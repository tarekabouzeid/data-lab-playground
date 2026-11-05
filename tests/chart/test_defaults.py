"""Default values must render the same resource set as chart 0.1.0 (Sandpiper upgrade creates/deletes nothing)."""
from conftest import REPO, render

GOLDEN = REPO / "tests" / "chart" / "golden" / "default-resources.txt"


def resource_set(docs):
    return sorted(f"{d['kind']}/{d['metadata']['name']}" for d in docs)


def test_default_resource_set_unchanged():
    assert resource_set(render()) == GOLDEN.read_text().split()
