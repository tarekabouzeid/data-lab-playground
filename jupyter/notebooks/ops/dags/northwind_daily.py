"""northwind_daily: Northwind Gear's daily pipeline, on Airflow 3 (stage 8 of the lab's data story).

  ingest_order_events -> dbt_build -> quality_checks -> publish_product_kpis
                                   -> wait_for_published

* ingest_order_events   Spark Connect: append the day's order events (lab_data.order_events(), one new `replay`) to
                        shop_bronze.order_events, the bronze table iceberg/04 streams into
* dbt_build             `dbt build` of the dbt project (seeds -> staging -> marts -> snapshots, with its tests) on Trino
* quality_checks        no orphan refunds, no events or orders for unknown orders/products, no negative amounts,
                        one row per order in the marts
* publish_product_kpis  shop_gold.product_kpis: orders, gross sales, refunded amount and refund rate per product
                        (the semantic model's refund_rate definition), stamped with the run's logical date
* wait_for_published    the Kubeflow Semantic Operator re-checks the SemanticModel against the rebuilt tables; wait
                        until its Published condition is True (skipped where there is no semantic layer)

Needs the lab Airflow image (airflow/Dockerfile: pyspark-client + dbt) and `airflow.labPipelines.enabled`, which sets
SPARK_REMOTE, TRINO_HOST/TRINO_PORT and SEMANTIC_MODEL_RESOURCE and lets the pod read its SemanticModels. The
workstation profile does both and mounts jupyter/notebooks as the DAG volume (subPath ops/dags), so lab_data.py and
the dbt project sit two folders up from this file. ops/01_pipeline_agent.ipynb asks the Airflow MCP server about runs.
"""
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import requests

NOTEBOOKS = Path(__file__).resolve().parents[2]          # jupyter/notebooks: lab_data.py, lab_utils.py, dbt/
DBT_PROJECT = NOTEBOOKS / "dbt" / "lakehouse_demo"
TRINO = f"http://{os.environ.get('TRINO_HOST', 'trino')}:{os.environ.get('TRINO_PORT', '8080')}"
HEADERS = {"X-Trino-User": "airflow", "X-Trino-Catalog": "iceberg", "X-Trino-Source": "northwind_daily"}

EVENTS_TABLE = "shop_bronze.order_events"                  # same schema as iceberg/04
EVENTS_DDL = """CREATE TABLE IF NOT EXISTS {t} (event_id BIGINT, replay BIGINT, order_id BIGINT, customer_id BIGINT,
                product_id BIGINT, quantity INT, order_ts TIMESTAMP, status STRING, ingested_at TIMESTAMP)
                USING iceberg TBLPROPERTIES ('format-version' = '2')"""

# name -> query returning one number that must be 0
CHECKS = {
    "orphan refunds": """SELECT count(*) FROM iceberg.dbt_demo_raw.raw_refunds r
                         LEFT JOIN iceberg.dbt_demo_raw.raw_orders o ON o.order_id = r.order_id WHERE o.order_id IS NULL""",
    "events for unknown orders": """SELECT count(*) FROM iceberg.shop_bronze.order_events e
                         LEFT JOIN iceberg.dbt_demo_raw.raw_orders o ON o.order_id = e.order_id WHERE o.order_id IS NULL""",
    "orders with an unknown product": """SELECT count(*) FROM iceberg.dbt_demo.fct_orders o
                         LEFT JOIN iceberg.dbt_demo.dim_products p ON p.product_id = o.product_id WHERE p.product_id IS NULL""",
    "negative amounts": """SELECT (SELECT count(*) FROM iceberg.dbt_demo_raw.raw_payments WHERE amount < 0)
                                + (SELECT count(*) FROM iceberg.dbt_demo_raw.raw_refunds WHERE amount < 0)""",
    "duplicate orders in the marts": """SELECT count(*) - count(DISTINCT order_id) FROM iceberg.dbt_demo.fct_orders""",
}

GOLD = [
    "CREATE SCHEMA IF NOT EXISTS iceberg.shop_gold",
    """CREATE OR REPLACE TABLE iceberg.shop_gold.product_kpis AS
       WITH sales AS (
         SELECT p.product_name AS product, count(*) AS orders, sum(o.amount) AS gross_sales
         FROM iceberg.dbt_demo.fct_orders o JOIN iceberg.dbt_demo.dim_products p ON p.product_id = o.product_id
         GROUP BY 1),
       refunds AS (
         SELECT p.product_name AS product, sum(r.amount) AS refunded_amount
         FROM iceberg.dbt_demo.fct_refunds r
         JOIN iceberg.dbt_demo.fct_orders o ON o.order_id = r.order_id
         JOIN iceberg.dbt_demo.dim_products p ON p.product_id = o.product_id
         GROUP BY 1)
       SELECT DATE '{ds}' AS as_of, s.product, s.orders, s.gross_sales,
              coalesce(r.refunded_amount, 0) AS refunded_amount,
              coalesce(r.refunded_amount, 0) / nullif(s.gross_sales, 0) AS refund_rate
       FROM sales s LEFT JOIN refunds r ON r.product = s.product""",
]

try:
    from airflow.sdk.exceptions import AirflowSkipException as Skip
except ImportError:      # outside Airflow (unit tests)
    class Skip(Exception):
        pass


def _lab():
    """lab_utils / lab_data from the notebooks folder (the DAG volume on the chart, the repo elsewhere)."""
    if str(NOTEBOOKS) not in sys.path:
        sys.path.insert(0, str(NOTEBOOKS))
    import lab_data
    import lab_utils
    return lab_utils, lab_data


def trino_sql(sql: str) -> list:
    """Run one statement through Trino's HTTP protocol and return all rows (raises on a query error)."""
    res = requests.post(f"{TRINO}/v1/statement", data=sql.encode(), headers=HEADERS, timeout=60)
    res.raise_for_status()
    body, rows = res.json(), []
    while True:
        if body.get("error"):
            raise RuntimeError(f"Trino: {body['error'].get('message')}\n{sql}")
        rows.extend(body.get("data") or [])
        if not body.get("nextUri"):
            return rows
        if not body.get("data"):
            time.sleep(0.2)
        res = requests.get(body["nextUri"], headers=HEADERS, timeout=60)
        res.raise_for_status()
        body = res.json()


def order_event_rows(events: list, replay: int) -> list:
    """One replay of the order log as rows of shop_bronze.order_events (event ids continue where iceberg/04's do)."""
    n = len(events)
    return [{"event_id": replay * n + i, "replay": replay, "order_id": e["order_id"], "customer_id": e["customer_id"],
             "product_id": e["product_id"], "quantity": e["quantity"], "order_ts": e["order_ts"], "status": e["status"]}
            for i, e in enumerate(events)]


def ingest_order_events(spark=None) -> dict:
    """Spark Connect: append one replay of the order events to the bronze table."""
    lab_utils, lab_data = _lab()
    own = spark is None
    spark = spark or lab_utils.get_spark("northwind_daily")
    try:
        from pyspark.sql import functions as F

        table = f"{lab_utils.CATALOG}.{EVENTS_TABLE}"
        spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {lab_utils.CATALOG}.shop_bronze")
        spark.sql(EVENTS_DDL.format(t=table))
        replay = spark.sql(f"SELECT coalesce(max(replay), -1) + 1 AS r FROM {table}").collect()[0][0]
        rows = order_event_rows(lab_data.order_events(), replay)
        schema = ("event_id long, replay long, order_id long, customer_id long, product_id long, quantity int, "
                  "order_ts string, status string")
        (spark.createDataFrame(rows, schema)
              .withColumn("order_ts", F.to_timestamp("order_ts"))
              .withColumn("ingested_at", F.current_timestamp())
              .writeTo(table).append())
        return {"table": table, "replay": replay, "events": len(rows)}
    finally:
        if own:
            spark.stop()


def dbt_build(run=subprocess.run) -> str:
    """`dbt build` on Trino. Target and logs go to a temp folder: the project folder may be mounted read-only."""
    work = Path(tempfile.mkdtemp(prefix="northwind-dbt-"))
    cmd = [os.environ.get("DBT_BIN", "dbt"), "build", "--project-dir", str(DBT_PROJECT),
           "--profiles-dir", str(DBT_PROJECT), "--target-path", str(work / "target"), "--log-path", str(work / "logs")]
    res = run(cmd, capture_output=True, text=True, env={**os.environ, "DBT_SEND_ANONYMOUS_USAGE_STATS": "false"})
    print(res.stdout[-6000:])
    if res.returncode != 0:
        raise RuntimeError(f"dbt build failed (exit {res.returncode}):\n{res.stdout[-3000:]}\n{res.stderr[-2000:]}")
    done = [line for line in res.stdout.splitlines() if "Done." in line]
    return done[-1].strip() if done else "dbt build finished"


def quality_checks(run=trino_sql) -> dict:
    results = {name: run(sql)[0][0] for name, sql in CHECKS.items()}
    failed = {k: v for k, v in results.items() if v}
    if failed:
        raise ValueError(f"data quality checks failed: {failed}")
    return results


def publish_product_kpis(ds: str, run=trino_sql) -> list:
    for sql in GOLD:
        run(sql.format(ds=ds))
    return run("SELECT product, refund_rate FROM iceberg.shop_gold.product_kpis ORDER BY refund_rate DESC LIMIT 3")


def _k8s_semantic_model(name: str) -> dict:
    from kubernetes import client, config

    config.load_incluster_config()
    namespace = os.environ.get("DATALAB_NAMESPACE") or \
        Path("/var/run/secrets/kubernetes.io/serviceaccount/namespace").read_text().strip()
    return client.CustomObjectsApi().get_namespaced_custom_object(
        "semantic.ossie.io", "v1alpha1", namespace, "semanticmodels", name)


def wait_for_published(get_model=None, timeout_s: int = 600, poll_s: int = 10) -> dict:
    """Wait until the SemanticModel's Published condition is True (the operator re-checks it against the tables)."""
    name = os.environ.get("SEMANTIC_MODEL_RESOURCE")
    if not name:
        raise Skip("no semantic layer in this deployment (semantic.enabled and airflow.labPipelines.enabled)")
    get_model = get_model or _k8s_semantic_model
    deadline = time.time() + timeout_s
    while True:
        conditions = {c.get("type"): c for c in (get_model(name).get("status") or {}).get("conditions") or []}
        published = conditions.get("Published", {})
        if published.get("status") == "True":
            return {"model": name, "published": True, "reason": published.get("reason")}
        if time.time() >= deadline:
            raise TimeoutError(f"SemanticModel {name}: Published={published.get('status')!r} after {timeout_s}s "
                               f"({published.get('reason')}: {published.get('message')})")
        time.sleep(poll_s)


try:
    from airflow.sdk import dag, task
except ImportError:      # imported outside Airflow (unit tests, notebooks): the task functions above still work
    dag = None

if dag is not None:
    @dag(dag_id="northwind_daily", schedule="@daily", start_date=datetime(2026, 9, 1), catchup=False,
         is_paused_upon_creation=False, tags=["northwind", "lab"], doc_md=__doc__)
    def northwind_daily():
        @task
        def ingest():
            return ingest_order_events()

        @task
        def build(_ingested):
            return dbt_build()

        @task
        def checks(_built):
            return quality_checks()

        @task
        def gold(_checks, ds=None):
            return publish_product_kpis(ds)

        @task
        def published(_built):
            return wait_for_published()

        built = build(ingest())
        gold(checks(built))
        published(built)

    northwind_daily()
