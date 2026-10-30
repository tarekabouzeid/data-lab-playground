"""ops/dags/northwind_daily.py: the tasks' SQL and control flow, against a fake Trino (Airflow is not needed)."""
import importlib.util
import re
from pathlib import Path

import pytest

DAG_FILE = Path(__file__).resolve().parents[2] / "jupyter" / "notebooks" / "ops" / "dags" / "northwind_daily.py"
spec = importlib.util.spec_from_file_location("northwind_daily", DAG_FILE)
dagmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dagmod)


class FakeTrino:
    def __init__(self, answers=None, missing=()):
        self.sql, self.answers, self.missing = [], answers or {}, missing

    def __call__(self, sql):
        self.sql.append(sql)
        for table in self.missing:
            if table in sql:
                raise RuntimeError(f"Trino: Table '{table}' does not exist")
        for pattern, rows in self.answers.items():
            if pattern in sql:
                return rows
        return [[0]]


def test_dag_id_and_trino_from_env():
    src = DAG_FILE.read_text()
    assert 'dag_id="northwind_daily"' in src
    assert "os.environ.get('TRINO_HOST', 'trino')" in src      # Compose default, chart sets TRINO_HOST


def test_check_inputs_names_the_missing_table():
    with pytest.raises(RuntimeError, match="dbt_demo.fct_orders is missing"):
        dagmod.check_inputs(FakeTrino(missing=("iceberg.dbt_demo.fct_orders",)))


def test_bronze_is_an_idempotent_merge():
    fake = FakeTrino(answers={"SELECT count(*) FROM iceberg.shop_bronze.orders_current": [[60]]})
    assert dagmod.upsert_bronze_orders(fake) == 60
    merge = next(s for s in fake.sql if s.lstrip().startswith("MERGE"))
    assert "WHEN MATCHED AND t.status <> s.status THEN UPDATE" in merge and "WHEN NOT MATCHED THEN INSERT" in merge
    assert all("IF NOT EXISTS" in s for s in fake.sql if s.startswith("CREATE"))


def test_quality_checks_fail_on_any_violation():
    assert set(dagmod.quality_checks(FakeTrino())) == set(dagmod.CHECKS)
    with pytest.raises(ValueError, match="orphan refunds"):
        dagmod.quality_checks(FakeTrino(answers={"raw_refunds r": [[2]]}))


def test_gold_uses_the_semantic_refund_rate_and_the_run_date():
    fake = FakeTrino(answers={"ORDER BY refund_rate": [["tablet", 0.625]]})
    assert dagmod.publish_product_kpis("2026-10-30", fake) == [["tablet", 0.625]]
    ctas = next(s for s in fake.sql if "CREATE OR REPLACE TABLE" in s)
    assert "DATE '2026-10-30'" in ctas
    assert re.search(r"refunded_amount, 0\) / nullif\(s.gross_sales, 0\)", ctas)   # SUM(refunds) / SUM(orders)


def test_trino_sql_follows_next_uri(monkeypatch):
    pages = [{"nextUri": "http://t/2"}, {"data": [[1]], "nextUri": "http://t/3"}, {"data": [[2]]}]

    class Res:
        def __init__(self, body):
            self.body = body

        def raise_for_status(self):
            pass

        def json(self):
            return self.body

    monkeypatch.setattr(dagmod.requests, "post", lambda *a, **k: Res(pages[0]))
    it = iter(pages[1:])
    monkeypatch.setattr(dagmod.requests, "get", lambda *a, **k: Res(next(it)))
    monkeypatch.setattr(dagmod.time, "sleep", lambda s: None)
    assert dagmod.trino_sql("SELECT 1") == [[1], [2]]
    monkeypatch.setattr(dagmod.requests, "post", lambda *a, **k: Res({"error": {"message": "boom"}}))
    with pytest.raises(RuntimeError, match="boom"):
        dagmod.trino_sql("SELECT 1")
