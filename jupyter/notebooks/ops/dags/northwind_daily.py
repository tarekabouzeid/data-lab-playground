"""northwind_daily: Northwind Gear's daily pipeline, on Airflow 3 (stage 8 of the lab's data story).

  check_inputs -> upsert_bronze_orders -> quality_checks -> publish_product_kpis

* check_inputs          the seeds (dbt_demo_raw) and the dbt marts (dbt_demo) exist; run dbt/01 once if not
* upsert_bronze_orders  MERGE the raw orders into shop_bronze.orders_current: new orders are inserted, status changes
                        (placed -> shipped -> returned) update the row; one Iceberg snapshot per run
* quality_checks        no orphan refunds, no unknown products, no negative amounts, one row per order
* publish_product_kpis  shop_gold.product_kpis: orders, gross sales, refunded amount and refund rate per product
                        (the semantic model's refund_rate definition), stamped with the run's logical date

Every task is SQL on Trino over HTTP (`requests` ships with Airflow), so the stock apache/airflow image runs it.
Trino's address comes from TRINO_HOST / TRINO_PORT (the workstation profile sets them via airflow.extraEnv); the
default is the Docker Compose host name. The workstation profile mounts this folder as the DAG folder
(airflow.dags.pvc.existingClaim = the notebooks claim, subPath ops/dags). ops/01_pipeline_agent.ipynb asks the
Airflow MCP server about the runs.
"""
import os
import time
from datetime import datetime

import requests

TRINO = f"http://{os.environ.get('TRINO_HOST', 'trino')}:{os.environ.get('TRINO_PORT', '8080')}"
HEADERS = {"X-Trino-User": "airflow", "X-Trino-Catalog": "iceberg", "X-Trino-Source": "northwind_daily"}

INPUTS = ["iceberg.dbt_demo_raw.raw_orders", "iceberg.dbt_demo_raw.raw_refunds", "iceberg.dbt_demo_raw.raw_payments",
          "iceberg.dbt_demo_raw.raw_products", "iceberg.dbt_demo.fct_orders", "iceberg.dbt_demo.fct_refunds",
          "iceberg.dbt_demo.dim_products"]

BRONZE = [
    "CREATE SCHEMA IF NOT EXISTS iceberg.shop_bronze",
    """CREATE TABLE IF NOT EXISTS iceberg.shop_bronze.orders_current (
         order_id bigint, customer_id bigint, product_id bigint, quantity integer, order_ts timestamp(6),
         status varchar, loaded_at timestamp(6))""",
    """MERGE INTO iceberg.shop_bronze.orders_current t
       USING (SELECT CAST(order_id AS bigint) AS order_id, CAST(customer_id AS bigint) AS customer_id,
                     CAST(product_id AS bigint) AS product_id, CAST(quantity AS integer) AS quantity,
                     CAST(order_ts AS timestamp(6)) AS order_ts, status
              FROM iceberg.dbt_demo_raw.raw_orders) s
       ON t.order_id = s.order_id
       WHEN MATCHED AND t.status <> s.status THEN UPDATE SET status = s.status, loaded_at = CAST(current_timestamp AS timestamp(6))
       WHEN NOT MATCHED THEN INSERT VALUES (s.order_id, s.customer_id, s.product_id, s.quantity, s.order_ts, s.status,
                                            CAST(current_timestamp AS timestamp(6)))""",
]

# name -> query returning one number that must be 0
CHECKS = {
    "orphan refunds": """SELECT count(*) FROM iceberg.dbt_demo_raw.raw_refunds r
                         LEFT JOIN iceberg.dbt_demo_raw.raw_orders o ON o.order_id = r.order_id WHERE o.order_id IS NULL""",
    "orders with an unknown product": """SELECT count(*) FROM iceberg.shop_bronze.orders_current o
                         LEFT JOIN iceberg.dbt_demo_raw.raw_products p ON p.product_id = o.product_id WHERE p.product_id IS NULL""",
    "negative amounts": """SELECT (SELECT count(*) FROM iceberg.dbt_demo_raw.raw_payments WHERE amount < 0)
                                + (SELECT count(*) FROM iceberg.dbt_demo_raw.raw_refunds WHERE amount < 0)""",
    "duplicate orders in bronze": """SELECT count(*) FROM (SELECT order_id FROM iceberg.shop_bronze.orders_current
                                     GROUP BY order_id HAVING count(*) > 1)""",
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


def check_inputs(run=trino_sql) -> dict:
    counts = {}
    for table in INPUTS:
        try:
            counts[table] = run(f"SELECT count(*) FROM {table}")[0][0]
        except RuntimeError as exc:
            raise RuntimeError(f"{table} is missing: run the dbt seeds and models first (notebook dbt/01)") from exc
    return counts


def upsert_bronze_orders(run=trino_sql) -> int:
    for sql in BRONZE:
        run(sql)
    return run("SELECT count(*) FROM iceberg.shop_bronze.orders_current")[0][0]


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


try:
    from airflow.sdk import dag, task
except ImportError:      # imported outside Airflow (unit tests, notebooks): the task functions above still work
    dag = None

if dag is not None:
    @dag(dag_id="northwind_daily", schedule="@daily", start_date=datetime(2026, 9, 1), catchup=False,
         is_paused_upon_creation=False, tags=["northwind", "lab"], doc_md=__doc__)
    def northwind_daily():
        @task
        def inputs():
            return check_inputs()

        @task
        def bronze(_counts):
            return upsert_bronze_orders()

        @task
        def checks(_rows):
            return quality_checks()

        @task
        def gold(_checks, ds=None):
            return publish_product_kpis(ds)

        gold(checks(bronze(inputs())))

    northwind_daily()
