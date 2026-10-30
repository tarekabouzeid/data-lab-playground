"""ci/workstation-values.yaml: the minikube lab profile keeps every Sandpiper guarantee (namespaced, no hostPath)."""
from conftest import by_kind, render, workload

API = ["gateway.networking.k8s.io/v1/Gateway", "semantic.ossie.io/v1alpha1/SemanticModel"]


def ws():
    return render(profile="workstation-values.yaml", release="datalab", namespace="datalab", api_versions=API)


def test_namespaced_and_no_host_path():
    docs = ws()
    assert not {d["kind"] for d in docs} & {"ClusterRole", "ClusterRoleBinding", "GatewayClass", "PersistentVolume",
                                            "CustomResourceDefinition", "Namespace"}
    assert "hostPath" not in str(docs)


def test_notebooks_from_the_host_folder_claim():
    vols = workload(ws(), "jupyter")["spec"]["template"]["spec"]["volumes"]
    assert {"name": "data", "persistentVolumeClaim": {"claimName": "datalab-notebooks"}} in vols


def test_urls_match_the_old_minikube_lab():
    hosts = {r["spec"]["hostnames"][0] for r in by_kind(ws(), "HTTPRoute")}
    assert {"jupyter.datalab.localhost", "trino.datalab.localhost", "headlamp.datalab.localhost"} <= hosts


def test_tool_calling_model_is_pulled():
    script = workload(ws(), "ollama")["spec"]["template"]["spec"]["containers"][0]["command"][2]
    assert "qwen3:4b" in script


def test_semantic_and_spark_apps_on():
    docs = ws()
    assert by_kind(docs, "SemanticModel")
    assert "datalab-data-lab-spark-apps" in {s["metadata"]["name"] for s in by_kind(docs, "ServiceAccount")}
