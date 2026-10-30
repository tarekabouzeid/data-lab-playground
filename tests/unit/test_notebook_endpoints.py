"""Notebooks run on Compose and on the chart, so service URLs live only in lab_utils.py (env with Compose default)."""
import json
import re
from pathlib import Path

NB = Path(__file__).resolve().parents[2] / "jupyter" / "notebooks"
HOST = re.compile(r"https?://(phoenix|ollama|qdrant|trino|hive-metastore|seaweedfs)[:/]|host\s*=\s*[\"']trino[\"']")


def code_cells():
    for f in sorted(NB.rglob("*.ipynb")):
        for i, c in enumerate(json.loads(f.read_text())["cells"]):
            if c["cell_type"] == "code":
                yield f.relative_to(NB), i, "".join(c["source"])


def test_no_hardcoded_service_hosts_in_notebooks():
    hits = [f"{f} cell {i}: {m.group(0)}" for f, i, src in code_cells() for m in [HOST.search(src)] if m]
    assert not hits, "use the lab_utils endpoint constants instead:\n" + "\n".join(hits)


def test_e2e_script_takes_endpoints_from_env():
    src = (NB.parents[1] / "tests" / "e2e" / "e2e_lakehouse.py").read_text()
    hard = [ln.strip() for ln in src.splitlines()
            if re.search(r"(thrift|https?)://(hive-metastore|trino|seaweedfs)[:/]", ln) and "os.environ.get(" not in ln]
    assert not hard, "e2e_lakehouse.py must read service URLs from env (Compose host only as the default):\n" + "\n".join(hard)
