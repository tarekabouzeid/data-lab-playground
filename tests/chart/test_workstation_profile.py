"""ci/workstation-values.yaml: the minikube lab profile keeps every Sandpiper guarantee (namespaced, no hostPath)."""
import re

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
    pulled = re.search(r"for m in (.*?); do", script).group(1).split()
    assert "qwen3:4b-instruct" in pulled
    assert "qwen3:4b" not in pulled   # Qwen3-4B-Thinking-2507: thinking-only, never reaches a tool call


def test_semantic_and_spark_apps_on():
    docs = ws()
    assert by_kind(docs, "SemanticModel")
    assert "datalab-data-lab-spark-apps" in {s["metadata"]["name"] for s in by_kind(docs, "ServiceAccount")}


def _render_two(values=None):
    import json
    import subprocess

    from conftest import CHART, KUBE_VERSION
    cmd = ["helm", "template", "datalab", str(CHART), "-n", "datalab", "--kube-version", KUBE_VERSION,
           "-f", str(CHART / "ci" / "workstation-values.yaml"), "-f", str(CHART / "ci" / "workstation-nogpu-values.yaml"),
           "-f", "-"] + [x for a in API for x in ("--api-versions", a)]
    p = subprocess.run(cmd, input=json.dumps(values or {}), capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    import yaml
    return [d for d in yaml.safe_load_all(p.stdout) if d]


def test_nogpu_profile_has_no_ollama_and_llm_off():
    import pytest
    if __import__("shutil").which("helm") is None:
        pytest.skip("helm is not installed")
    docs = _render_two()
    assert not [d for d in docs if d.get("metadata", {}).get("labels", {}).get("app.kubernetes.io/component") == "ollama"]
    from conftest import configmap
    assert configmap(docs, "datalab-data-lab-endpoints")["LLM_PROVIDER"] == "none"


def test_nogpu_profile_with_an_openai_compatible_endpoint():
    import pytest
    if __import__("shutil").which("helm") is None:
        pytest.skip("helm is not installed")
    from conftest import configmap, env_of
    docs = _render_two({"llm": {"provider": "openai", "baseUrl": "https://llm.example.com/v1", "chatModel": "a",
                                "toolModel": "b", "embeddingModel": "e", "auth": {"existingSecret": "llm-key"}}})
    assert env_of(docs, "jupyter")["LLM_API_KEY"]["valueFrom"]["secretKeyRef"] == {"name": "llm-key", "key": "api-key"}
    assert configmap(docs, "datalab-data-lab-endpoints")["LLM_EMBED_MODEL"] == "e"
