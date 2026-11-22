"""SparkApplication service account (Spark Operator, optional) and the 19-check lakehouse helm test."""
from conftest import CHART, REPO, by_kind, configmap, render

APPS = {"spark": {"applications": {"enabled": True}}}


def names(docs, kind):
    return {d["metadata"]["name"] for d in by_kind(docs, kind)}


def test_spark_apps_off_by_default():
    docs = render()
    assert "r-data-lab-spark-apps" not in names(docs, "ServiceAccount")
    assert "SPARK_APPLICATION_SERVICE_ACCOUNT" not in configmap(docs, "r-data-lab-endpoints")


def test_spark_apps_rbac():
    docs = render(APPS)
    assert "r-data-lab-spark-apps" in names(docs, "ServiceAccount")
    [role] = [r for r in by_kind(docs, "Role") if r["metadata"]["name"] == "r-data-lab-spark-apps"]
    assert {"pods", "services", "configmaps", "persistentvolumeclaims"} <= set(role["rules"][0]["resources"])
    [rb] = [r for r in by_kind(docs, "RoleBinding") if r["metadata"]["name"] == "r-data-lab-spark-apps"]
    assert rb["roleRef"]["kind"] == "Role"
    assert rb["subjects"] == [{"kind": "ServiceAccount", "name": "r-data-lab-spark-apps", "namespace": "ns"}]
    assert not by_kind(docs, "ClusterRole") and not by_kind(docs, "ClusterRoleBinding")
    data = configmap(docs, "r-data-lab-endpoints")
    assert data["SPARK_APPLICATION_SERVICE_ACCOUNT"] == "r-data-lab-spark-apps"
    assert data["SPARK_DEFAULTS_CONFIGMAP"] == "r-data-lab-spark-connect"


def lakehouse_pod(docs):
    return next(p for p in by_kind(docs, "Pod") if p["metadata"]["name"] == "r-data-lab-test-lakehouse")


def test_lakehouse_test_off_by_default():
    assert "r-data-lab-test-lakehouse" not in names(render(), "Pod")


def test_lakehouse_test_pod():
    docs = render({"tests": {"lakehouseE2e": {"enabled": True}}})
    pod = lakehouse_pod(docs)
    assert pod["metadata"]["annotations"]["helm.sh/hook"] == "test"
    script = configmap(docs, "r-data-lab-test-lakehouse")["e2e_lakehouse.py"]
    assert script == (CHART / "files" / "e2e_lakehouse.py").read_text()
    env = {e["name"]: e for e in pod["spec"]["containers"][0]["env"]}
    assert env["TRINO_URL"]["value"] == "http://r-data-lab-trino.ns.svc:8080"
    assert ";authorization=Bearer $(SPARK_CONNECT_TOKEN)" in env["SPARK_REMOTE"]["value"]
    assert env["E2E_WAREHOUSE_BUCKET"]["value"] == "warehouse"
    assert env["ICEBERG_REST_URI"]["value"] == "http://r-data-lab-hms.ns.svc:9084/iceberg"


def test_lakehouse_test_passes_helm_lint():
    import subprocess

    p = subprocess.run(["helm", "lint", "--strict", str(REPO / "charts" / "data-lab"), "--kube-version", "1.34.0",
                        "--set", "tests.lakehouseE2e.enabled=true,spark.applications.enabled=true"], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr


def test_iceberg_roundtrip_uses_a_fresh_namespace_each_run():
    """HMS 4.2.1 + SeaweedFS cannot re-create a dropped namespace (the managed/<ns>.db marker stays; AGENTS.md
    pitfall 18). A fixed `helm_test` namespace made every `helm test` after the first one fail."""
    pod = next(d for d in by_kind(render(), "Pod") if d["metadata"]["name"] == "r-data-lab-test-iceberg")
    script = pod["spec"]["containers"][0]["command"][-1]
    assert 'f"helm_test_{uuid.uuid4().hex[:8]}"' in script
    assert '"helm_test"' not in script
