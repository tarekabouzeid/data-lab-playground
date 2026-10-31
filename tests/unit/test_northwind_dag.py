"""ops/dags/northwind_daily.py: the tasks' SQL and control flow, against a fake Trino (Airflow is not needed)."""
import importlib.util
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "jupyter" / "notebooks"))
import lab_data  # noqa: E402

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
    assert "os.environ.get('TRINO_HOST', 'trino')" in src      # Compose default, airflow.labPipelines sets TRINO_HOST
    assert dagmod.NOTEBOOKS.name == "notebooks" and (dagmod.DBT_PROJECT / "dbt_project.yml").exists()


def test_one_replay_of_the_order_log():
    events = lab_data.order_events()
    rows = dagmod.order_event_rows(events, replay=3)
    assert len(rows) == len(events) == 60
    assert [r["event_id"] for r in rows] == list(range(180, 240))      # continues iceberg/04's ids (replay * 60 + seq)
    assert {r["replay"] for r in rows} == {3}
    assert [r["order_id"] for r in rows] == [e["order_id"] for e in events]


def test_dbt_build_writes_outside_the_project_and_fails_loudly():
    calls = []

    class Res:
        def __init__(self, rc, out):
            self.returncode, self.stdout, self.stderr = rc, out, ""

    def run(cmd, **kw):
        calls.append((cmd, kw["env"]))
        return Res(0, "...\nDone. PASS=42 WARN=0 ERROR=0 SKIP=0 TOTAL=42\n")

    assert dagmod.dbt_build(run) == "Done. PASS=42 WARN=0 ERROR=0 SKIP=0 TOTAL=42"
    cmd, env = calls[0]
    assert cmd[1] == "build" and cmd[cmd.index("--project-dir") + 1] == str(dagmod.DBT_PROJECT)
    assert not cmd[cmd.index("--target-path") + 1].startswith(str(dagmod.DBT_PROJECT))   # project may be read-only
    assert env["DBT_SEND_ANONYMOUS_USAGE_STATS"] == "false"
    with pytest.raises(RuntimeError, match="dbt build failed"):
        dagmod.dbt_build(lambda cmd, **kw: Res(1, "ERROR=1"))


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


def test_wait_for_published(monkeypatch):
    monkeypatch.delenv("SEMANTIC_MODEL_RESOURCE", raising=False)
    with pytest.raises(dagmod.Skip):
        dagmod.wait_for_published(lambda name: {})
    monkeypatch.setenv("SEMANTIC_MODEL_RESOURCE", "datalab-data-lab-dbt-demo")
    states = iter([{}, {"status": {"conditions": [{"type": "Published", "status": "False", "reason": "DriftDetected"}]}},
                   {"status": {"conditions": [{"type": "Published", "status": "True", "reason": "Served"}]}}])
    seen = []
    out = dagmod.wait_for_published(lambda name: (seen.append(name), next(states))[1], timeout_s=5, poll_s=0)
    assert out == {"model": "datalab-data-lab-dbt-demo", "published": True, "reason": "Served"}
    assert seen == ["datalab-data-lab-dbt-demo"] * 3
    stuck = {"status": {"conditions": [{"type": "Published", "status": "False", "reason": "DriftDetected", "message": "x"}]}}
    with pytest.raises(TimeoutError, match="DriftDetected"):
        dagmod.wait_for_published(lambda name: stuck, timeout_s=0, poll_s=0)


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
