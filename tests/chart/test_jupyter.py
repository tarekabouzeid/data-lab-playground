"""Jupyter on the chart gets every endpoint the notebooks read (lab_utils.py), with credentials from Secrets."""
import re

from conftest import REPO, by_kind, configmap, env_of, render, workload

ON = {"jupyter": {"enabled": True}}
# Every service the notebooks talk to, so every env var lab_utils reads has a source.
ALL = {k: {"enabled": True} for k in ("jupyter", "ollama", "qdrant", "phoenix")}
LAB_UTILS = (REPO / "jupyter" / "notebooks" / "lab_utils.py").read_text()


def names_read_by_lab_utils():
    return set(re.findall(r"os\.environ\.get\(\s*[\"']([A-Z_]+)[\"']", LAB_UTILS))


def visible_env(docs):
    """Env names the Jupyter container sees: its env plus the endpoints ConfigMap (envFrom)."""
    return set(env_of(docs, "jupyter")) | set(configmap(docs, "r-data-lab-endpoints"))


def test_jupyter_env_matches_lab_utils():
    docs = render(ALL)
    missing = names_read_by_lab_utils() - visible_env(docs) - {"SPARK_REMOTE"}
    assert not missing, f"lab_utils.py reads env vars the chart does not set: {sorted(missing)}"
    assert "SPARK_REMOTE" in env_of(docs, "jupyter")


def test_spark_remote_has_bearer():
    env = env_of(render(ON), "jupyter")
    assert ";authorization=Bearer $(SPARK_CONNECT_TOKEN)" in env["SPARK_REMOTE"]["value"]
    assert env["SPARK_CONNECT_TOKEN"]["valueFrom"]["secretKeyRef"]["name"] == "r-data-lab-spark-connect"


def test_s3_credentials_from_secret():
    env = env_of(render(ON), "jupyter")
    for k in ("S3_ACCESS_KEY", "S3_SECRET_KEY"):
        assert "secretKeyRef" in env[k]["valueFrom"], k


def test_existing_claim():
    docs = render({"jupyter": {"enabled": True, "persistence": {"existingClaim": "nb"}}})
    vols = workload(docs, "jupyter")["spec"]["template"]["spec"]["volumes"]
    assert {"name": "data", "persistentVolumeClaim": {"claimName": "nb"}} in vols
    assert not [p for p in by_kind(docs, "PersistentVolumeClaim") if p["metadata"]["name"] == "r-data-lab-jupyter"]
