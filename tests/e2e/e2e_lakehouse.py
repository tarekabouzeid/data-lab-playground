"""End-to-end lakehouse test: Spark + Iceberg + Hive Metastore + Trino + MinIO.

Run via tests/e2e/run-e2e.sh (submitted with spark-submit inside the
datalab-playground/spark image, on the platform's Docker network).

Covers:
  1. Spark  -> Iceberg table via HiveCatalog (HMS): create, insert, schema evolution
  2. Spark  -> Iceberg snapshots + time travel
  3. Trino  `iceberg` catalog reads the Spark-written table, $snapshots, FOR VERSION AS OF
  4. Trino  `iceberg` INSERT, read back by Spark (bidirectional)
  5. Spark  -> plain Parquet on MinIO; Trino `hive` external table over it
  6. Trino  `hive` managed CTAS table
  7. Trino  `lakehouse` catalog reads both the Hive and the Iceberg table
  8. Trino  cross-catalog JOIN hive x iceberg
"""

import json
import sys
import time
import urllib.request

from pyspark.sql import SparkSession

TRINO_URL = "http://trino:8080"
SCHEMA = "e2e"
ICEBERG_TABLE = f"iceberg_catalog.{SCHEMA}.sales"
PARQUET_PATH = f"s3a://warehouse/{SCHEMA}/transactions_parquet/"

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}", flush=True)
    if not cond:
        raise AssertionError(f"{name}: {detail}")


def trino(sql):
    """Minimal Trino REST client (no extra Python packages needed)."""
    req = urllib.request.Request(
        f"{TRINO_URL}/v1/statement",
        data=sql.encode(),
        headers={"X-Trino-User": "e2e", "Content-Type": "text/plain"},
        method="POST",
    )
    payload = json.load(urllib.request.urlopen(req, timeout=120))
    rows = []
    while True:
        if "error" in payload:
            err = payload["error"]
            raise RuntimeError(f"Trino error for [{sql}]: {err.get('message')}")
        rows.extend(payload.get("data", []))
        next_uri = payload.get("nextUri")
        if not next_uri:
            return rows
        time.sleep(0.05)
        payload = json.load(urllib.request.urlopen(next_uri, timeout=120))


def main():
    spark = (
        SparkSession.builder.appName("datalab-e2e")
        .config("spark.sql.extensions",
                "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
        .config("spark.sql.catalog.iceberg_catalog", "org.apache.iceberg.spark.SparkCatalog")
        .config("spark.sql.catalog.iceberg_catalog.type", "hive")
        .config("spark.sql.catalog.iceberg_catalog.uri", "thrift://hive-metastore:9083")
        .config("spark.sql.catalog.iceberg_catalog.warehouse", "s3a://warehouse/")
        .getOrCreate()
    )
    print(f"Spark {spark.version}", flush=True)
    check("spark version is 4.1.x", spark.version.startswith("4.1."), spark.version)

    # ---- clean slate -------------------------------------------------------
    for stmt in [
        f"DROP TABLE IF EXISTS hive.{SCHEMA}.transactions",
        f"DROP TABLE IF EXISTS hive.{SCHEMA}.city_summary",
    ]:
        try:
            trino(stmt)
        except RuntimeError as e:
            print(f"cleanup (ignored): {e}")
    spark.sql(f"DROP TABLE IF EXISTS {ICEBERG_TABLE} PURGE")
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS iceberg_catalog.{SCHEMA} "
              f"LOCATION 's3a://warehouse/{SCHEMA}.db'")

    # ---- 1. Spark -> Iceberg via HMS --------------------------------------
    rows = [
        (1, "smartphone", "Electronics", 699.99, "New York"),
        (2, "laptop", "Electronics", 1299.99, "California"),
        (3, "headphones", "Electronics", 199.99, "Texas"),
        (4, "tablet", "Electronics", 499.99, "Florida"),
        (5, "smartwatch", "Electronics", 299.99, "Washington"),
    ]
    df = spark.createDataFrame(rows, ["product_id", "product_name", "category", "price", "region"])
    (df.writeTo(ICEBERG_TABLE).using("iceberg")
       .tableProperty("format-version", "2")
       .tableProperty("write.parquet.compression-codec", "snappy")
       .create())
    check("spark: iceberg create via HMS",
          spark.table(ICEBERG_TABLE).count() == 5)

    spark.sql(f"INSERT INTO {ICEBERG_TABLE} VALUES (6, 'camera', 'Electronics', 899.99, 'Oregon')")
    spark.sql(f"ALTER TABLE {ICEBERG_TABLE} ADD COLUMN discount DOUBLE")
    spark.sql(f"INSERT INTO {ICEBERG_TABLE} VALUES (7, 'keyboard', 'Electronics', 89.99, 'Oregon', 0.1)")
    check("spark: iceberg insert + schema evolution",
          spark.table(ICEBERG_TABLE).count() == 7
          and "discount" in spark.table(ICEBERG_TABLE).columns)

    # ---- 2. Snapshots / time travel ---------------------------------------
    snaps = spark.sql(f"SELECT snapshot_id FROM {ICEBERG_TABLE}.snapshots ORDER BY committed_at").collect()
    check("spark: 3 snapshots", len(snaps) == 3, str(len(snaps)))
    first = snaps[0]["snapshot_id"]
    tt = spark.sql(f"SELECT * FROM {ICEBERG_TABLE} VERSION AS OF {first}").count()
    check("spark: time travel to first snapshot", tt == 5, str(tt))

    # ---- 3. Trino iceberg catalog reads Spark table -----------------------
    cnt = trino(f"SELECT count(*) FROM iceberg.{SCHEMA}.sales")[0][0]
    check("trino iceberg: read spark-written table", cnt == 7, str(cnt))
    disc = trino(f"SELECT discount FROM iceberg.{SCHEMA}.sales WHERE product_id = 7")[0][0]
    check("trino iceberg: evolved column visible", abs(disc - 0.1) < 1e-9, str(disc))
    tsnaps = trino(f'SELECT snapshot_id FROM iceberg.{SCHEMA}."sales$snapshots" ORDER BY committed_at')
    check("trino iceberg: $snapshots", len(tsnaps) == 3, str(len(tsnaps)))
    tt = trino(f"SELECT count(*) FROM iceberg.{SCHEMA}.sales FOR VERSION AS OF {tsnaps[0][0]}")[0][0]
    check("trino iceberg: FOR VERSION AS OF", tt == 5, str(tt))

    # ---- 4. Trino writes, Spark reads -------------------------------------
    trino(f"INSERT INTO iceberg.{SCHEMA}.sales VALUES (8, 'monitor', 'Electronics', 349.99, 'Texas', 0.05)")
    spark.sql(f"REFRESH TABLE {ICEBERG_TABLE}")
    cnt = spark.table(ICEBERG_TABLE).count()
    check("spark: reads trino-written iceberg row", cnt == 8, str(cnt))

    # ---- 5. Plain Parquet + Trino hive external table ---------------------
    tx = spark.createDataFrame(
        [(f"C{i:03d}", ["laptop", "camera", "monitor", "tablet"][i % 4],
          float(10 + i), ["New York", "Texas", "Oregon"][i % 3])
         for i in range(100)],
        ["customer_id", "product", "total_amount", "city"],
    )
    tx.write.mode("overwrite").parquet(PARQUET_PATH)
    check("spark: parquet write to MinIO", spark.read.parquet(PARQUET_PATH).count() == 100)

    trino(f"CREATE SCHEMA IF NOT EXISTS hive.{SCHEMA}")
    trino(f"""
        CREATE TABLE hive.{SCHEMA}.transactions (
            customer_id VARCHAR, product VARCHAR, total_amount DOUBLE, city VARCHAR)
        WITH (external_location = '{PARQUET_PATH}', format = 'PARQUET')""")
    cnt = trino(f"SELECT count(*) FROM hive.{SCHEMA}.transactions")[0][0]
    check("trino hive: external parquet table", cnt == 100, str(cnt))

    # ---- 6. Trino hive managed CTAS ---------------------------------------
    trino(f"""
        CREATE TABLE hive.{SCHEMA}.city_summary WITH (format = 'PARQUET') AS
        SELECT city, count(*) AS n, sum(total_amount) AS revenue
        FROM hive.{SCHEMA}.transactions GROUP BY city""")
    n = trino(f"SELECT sum(n) FROM hive.{SCHEMA}.city_summary")[0][0]
    check("trino hive: managed CTAS", n == 100, str(n))

    # ---- 7. Lakehouse catalog reads both table types ----------------------
    a = trino(f"SELECT count(*) FROM lakehouse.{SCHEMA}.transactions")[0][0]
    b = trino(f"SELECT count(*) FROM lakehouse.{SCHEMA}.sales")[0][0]
    check("trino lakehouse: hive + iceberg tables", a == 100 and b == 8, f"{a}/{b}")

    # ---- 8. Cross-catalog JOIN --------------------------------------------
    joined = trino(f"""
        SELECT e.product_name, count(*) AS n
        FROM hive.{SCHEMA}.transactions t
        JOIN iceberg.{SCHEMA}.sales e ON lower(t.product) = lower(e.product_name)
        GROUP BY e.product_name ORDER BY e.product_name""")
    check("trino: cross-catalog hive x iceberg join",
          {r[0] for r in joined} == {"camera", "laptop", "monitor", "tablet"}, str(joined))

    spark.stop()


if __name__ == "__main__":
    completed = False
    try:
        main()
        completed = True
    except Exception as e:  # noqa: BLE001
        print(f"E2E FAILED: {type(e).__name__}: {e}", flush=True)
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\n==== E2E SUMMARY: {passed}/{len(results)} checks passed"
          f"{'' if completed else ' (aborted)'} ====", flush=True)
    sys.exit(0 if completed and passed == len(results) else 1)
