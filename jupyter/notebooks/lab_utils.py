"""Helpers shared by the technique notebooks (iceberg/, dbt/).

Works unchanged on both deployments of the lab:
  * Docker Compose: classic Spark. The session is created here with the Iceberg REST catalog; the master and the S3A
    settings come from $SPARK_HOME/conf/spark-defaults.conf in the Jupyter image.
  * Kubernetes: Spark Connect (SPARK_REMOTE is set). The Spark Connect server already has the catalog, the Iceberg SQL
    extensions and the S3A settings (see helm/datalab/values.yaml, spark.sparkConf). Do not call .master() here:
    pyspark 4.1 rejects it together with SPARK_REMOTE, and static confs cannot be set by a Connect client.

Imports of optional packages (boto3, trino, pyarrow) are lazy so the module imports anywhere.
"""
import os
from pathlib import Path

CATALOG = "iceberg_catalog"                 # Spark name of the HMS Iceberg REST catalog
TRINO_CATALOG = "iceberg"                   # the same catalog as seen by Trino
REST_URI = os.environ.get("ICEBERG_REST_URI", "http://hive-metastore:9084/iceberg")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "http://seaweedfs:8333")
S3_KEY = os.environ.get("S3_ACCESS_KEY", "seaweedadmin")
S3_SECRET = os.environ.get("S3_SECRET_KEY", "seaweedadmin123")
TRINO_HOST = os.environ.get("TRINO_HOST", "trino")
TRINO_PORT = int(os.environ.get("TRINO_PORT", "8080"))


def on_spark_connect() -> bool:
    return bool(os.environ.get("SPARK_REMOTE"))


def get_spark(app_name: str = "lab-notebook"):
    """A Spark session wired to the Iceberg catalog, on either deployment."""
    from pyspark.sql import SparkSession

    builder = SparkSession.builder.appName(app_name)
    if not on_spark_connect():
        builder = (
            builder.config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
            .config(f"spark.sql.catalog.{CATALOG}", "org.apache.iceberg.spark.SparkCatalog")
            .config(f"spark.sql.catalog.{CATALOG}.type", "rest")
            .config(f"spark.sql.catalog.{CATALOG}.uri", REST_URI)
            .config(f"spark.sql.catalog.{CATALOG}.io-impl", "org.apache.iceberg.hadoop.HadoopFileIO")
        )
    return builder.getOrCreate()


def trino(sql: str):
    """Run one statement on Trino (catalog `iceberg`) and return a pandas DataFrame."""
    import pandas as pd
    import trino as trino_client

    conn = trino_client.dbapi.connect(host=TRINO_HOST, port=TRINO_PORT, user="lab", catalog=TRINO_CATALOG)
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    cols = [c.name for c in (cur.description or [])]
    return pd.DataFrame(rows, columns=cols)


def s3_client():
    import boto3

    return boto3.client("s3", endpoint_url=S3_ENDPOINT, aws_access_key_id=S3_KEY,
                        aws_secret_access_key=S3_SECRET, region_name="us-east-1")


def parquet_footer(s3a_path: str):
    """Parquet footer metadata of an s3a:// object (downloads the file; use small files)."""
    import io

    import pyarrow.parquet as pq

    bucket, key = s3a_path.replace("s3a://", "").split("/", 1)
    data = s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()
    return pq.ParquetFile(io.BytesIO(data)).metadata


def try_sql(spark, sql: str, label: str | None = None):
    """Run SQL and print OK / the first line of the error. For 'does this work here?' probes."""
    try:
        rows = spark.sql(sql).collect()
        print(f"OK     {label or sql[:70]}")
        return rows
    except Exception as exc:  # noqa: BLE001 - we want to show whatever the engine says
        first = str(exc).strip().splitlines()[0][:230]
        print(f"FAILED {label or sql[:70]}\n       -> {first}")
        return None


# --------------------------------------------------------------------------------------------------------------------
# dbt (dbt-core + dbt-trino are installed in the Jupyter image; the project talks to the same Trino as trino() above).
# Never name a Python module or package in the notebooks tree "dbt_*": dbt imports every importable module with that prefix as a plugin.
# --------------------------------------------------------------------------------------------------------------------
DBT_PROJECT = Path(__file__).resolve().parent / "dbt" / "lakehouse_demo"


def dbt(*args: str, project: Path = DBT_PROJECT, show_log: bool = False):
    """Run one dbt command in-process (dbtRunner) and return a DataFrame with one row per node.

    Example: dbt("build", "--select", "fct_orders"). Errors of failed nodes are printed in full.
    With show_log=True dbt's own console log is shown as well.
    """
    import pandas as pd
    from dbt.cli.main import dbtRunner

    argv = list(args) + ["--project-dir", str(project), "--profiles-dir", str(project), "--log-path", str(project / "logs"),
                         "--no-use-colors"] + ([] if show_log else ["--quiet"])
    res = dbtRunner().invoke(argv)
    if res.exception is not None:
        raise RuntimeError(f"dbt {' '.join(args)} could not start: {res.exception}")
    if isinstance(res.result, list):                       # e.g. `dbt ls` prints its own listing; return it for programmatic use
        return res.result
    results = getattr(res.result, "results", None)
    if not results:
        print(f"dbt {' '.join(args)}: {'ok' if res.success else 'FAILED'}")
        return res
    rows = [{"node": r.node.name, "type": r.node.resource_type.value,
             "status": str(r.status.value if hasattr(r.status, "value") else r.status),
             "seconds": round(r.execution_time, 2), "message": (r.message or "")[:300]} for r in results]
    df = pd.DataFrame(rows)
    bad = df[df.status.isin(["error", "fail", "runtime error"])]
    print(f"dbt {' '.join(args)}: {len(df)} nodes, {int((df.status.isin(['success', 'pass'])).sum())} ok, {len(bad)} not ok")
    for _, r in bad.iterrows():
        print(f"  [{r.status}] {r.node}: {r.message}")
    return df


DBT_SCHEMAS = ("dbt_demo", "dbt_demo_raw", "dbt_demo_snapshots", "dbt_test__audit")


def reset_dbt_demo():
    """Drop every table and view the dbt demo created, for a clean start. The schemas themselves are kept on purpose:
    in this lab a schema that was dropped cannot be created again under the same name (Hive Metastore fails with
    'Unable to create database managed directory' because an empty managed/<name>.db directory stays in SeaweedFS)."""
    existing = set(trino("SHOW SCHEMAS").iloc[:, 0])
    dropped = 0
    for schema in DBT_SCHEMAS:
        if schema not in existing:
            continue
        objs = trino(f"SELECT table_name, table_type FROM information_schema.tables WHERE table_schema = '{schema}'")
        for name, kind in objs.itertuples(index=False):
            trino(f'DROP {"VIEW" if kind == "VIEW" else "TABLE"} IF EXISTS iceberg.{schema}."{name}"')
            dropped += 1
    print(f"dbt demo reset: {dropped} tables/views dropped (schemas kept)")
