"""Docker Compose is gone: the Helm chart is the only way to run the lab."""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GONE = ["docker-compose.yaml", "start-platform.sh", "trino", "seaweedfs", "tests/e2e",
        "hack/check-compose-parity.py", "jupyter/spark-defaults.conf", "spark/conf"]
DOCS = ["README.md", "AGENTS.md", "CLAUDE.md", "docs/WORKSTATION_TESTING.md", "docs/VERSIONS.md"]
ALLOWED = "Compose removed in chart 0.4.0"


def test_compose_files_are_gone():
    assert [p for p in GONE if (REPO / p).exists()] == []


def test_living_docs_do_not_mention_compose():
    files = [REPO / d for d in DOCS] + sorted((REPO / "jupyter/notebooks").rglob("*.ipynb"))
    bad = []
    for f in files:
        text = f.read_text().replace(ALLOWED, "")
        if re.search(r"docker[ -]compose|start-platform", text, re.I):
            bad.append(str(f.relative_to(REPO)))
    assert bad == []
