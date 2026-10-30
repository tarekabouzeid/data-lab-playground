"""hack/check-compose-parity.py: Compose Spark conf vs the chart's rendered conf, and the e2e script copy."""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "hack" / "check-compose-parity.py"
needs_helm = pytest.mark.skipif(shutil.which("helm") is None, reason="helm is not installed")


def run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, cwd=REPO)


@needs_helm
def test_parity_holds_on_the_branch():
    p = run()
    assert p.returncode == 0, p.stdout + p.stderr


@needs_helm
def test_new_compose_key_is_caught(tmp_path):
    conf = tmp_path / "spark-defaults.conf"
    conf.write_text((REPO / "spark/conf/spark-defaults.conf").read_text() + "spark.sql.shuffle.partitions=7\n")
    p = run("--compose-conf", str(conf))
    assert p.returncode == 1
    assert "spark.sql.shuffle.partitions" in p.stdout


@needs_helm
def test_e2e_copy_drift_is_caught(tmp_path):
    e2e = tmp_path / "e2e_lakehouse.py"
    e2e.write_text((REPO / "tests/e2e/e2e_lakehouse.py").read_text() + "# drift\n")
    p = run("--e2e", str(e2e))
    assert p.returncode == 1
    assert "e2e_lakehouse.py" in p.stdout


@needs_helm
def test_notebook_model_copy_drift_is_caught(tmp_path):
    copy = tmp_path / "dbt_demo_model.yaml"
    copy.write_text((REPO / "charts/data-lab/files/semantic/dbt-demo.yaml").read_text() + "# drift\n")
    p = run("--semantic-model", str(copy))
    assert p.returncode == 1
    assert "dbt_demo_model.yaml" in p.stdout
