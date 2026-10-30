"""Kubeflow Semantic Operator (manager + server) rendered by the chart, namespaced, on the release's Trino."""
import pytest
import yaml

from conftest import REPO, RenderError, by_kind, configmap, render, workload

API = ["semantic.ossie.io/v1alpha1/SemanticModel"]
ON = {"semantic": {"enabled": True}}


def server_config(docs):
    return yaml.safe_load(configmap(docs, "r-data-lab-semantic-server")["config.yaml"])


def manager_config(docs):
    return yaml.safe_load(configmap(docs, "r-data-lab-semantic-manager")["config.yaml"])


def test_off_by_default():
    docs = render(api_versions=API)
    assert not [d for d in docs if "semantic" in d["metadata"]["name"]]


def test_requires_crd():
    with pytest.raises(RenderError, match="hack/crds"):
        render(ON)


def test_requires_trino():
    with pytest.raises(RenderError, match="semantic.enabled=true requires trino.enabled=true"):
        render({**ON, "trino": {"enabled": False}}, api_versions=API)


def test_namespaced_only():
    kinds = {d["kind"] for d in render(ON, api_versions=API)}
    assert not kinds & {"ClusterRole", "ClusterRoleBinding", "CustomResourceDefinition"}


def test_manager_and_server_deployments():
    docs = render(ON, api_versions=API)
    m, s = workload(docs, "semantic-manager"), workload(docs, "semantic-server")
    assert m["spec"]["template"]["spec"]["containers"][0]["image"] == "public.ecr.aws/data-on-eks/semantic-operator/manager:v0.1.1"
    assert s["spec"]["template"]["spec"]["containers"][0]["image"] == "public.ecr.aws/data-on-eks/semantic-operator/server:v0.1.1"
    assert s["spec"]["replicas"] == 1
    assert m["spec"]["template"]["spec"]["serviceAccountName"] == "r-data-lab-semantic-manager"
    assert s["spec"]["template"]["spec"]["serviceAccountName"] == "r-data-lab-semantic-server"
    [svc] = [d for d in by_kind(docs, "Service") if d["metadata"]["name"] == "r-data-lab-semantic"]
    assert svc["spec"]["type"] == "ClusterIP" and svc["spec"]["ports"][0]["port"] == 8090


def test_engine_points_at_release_trino():
    docs = render(ON, api_versions=API)
    for cfg, user in ((server_config(docs), "semantic-server"), (manager_config(docs), "semantic-manager")):
        assert cfg["engine"]["dialect"] == "trino"
        assert cfg["engine"]["connection"] == {"host": "r-data-lab-trino.ns.svc", "port": 8080, "user": user}


def test_manager_scope_is_the_release_namespace():
    cfg = manager_config(render(ON, api_versions=API))
    assert cfg["cache"]["watchNamespaces"] == ["ns"]
    assert cfg["leaderElection"] == {"leaderElect": False}   # one replica, Recreate: no Lease left after uninstall
    assert cfg["controller"]["resyncPeriod"] == "30s"
    assert server_config(render(ON, api_versions=API))["store"]["watchNamespace"] == "ns"


def test_header_auth_dev_only():
    assert server_config(render(ON, api_versions=API))["auth"] == {"mode": "header"}
    with pytest.raises(RenderError, match="jwksURL"):
        render({"semantic": {"enabled": True, "server": {"auth": {"mode": "jwt"}}}}, api_versions=API)
    jwt = {"semantic": {"enabled": True, "server": {"auth": {"mode": "jwt", "jwt": {"jwksURL": "https://idp/jwks", "issuer": "iss"}}}}}
    auth = server_config(render(jwt, api_versions=API))["auth"]
    assert auth["mode"] == "jwt" and auth["jwksURL"] == "https://idp/jwks" and auth["issuer"] == "iss"


def test_rbac_rules():
    roles = {r["metadata"]["name"]: r["rules"] for r in by_kind(render(ON, api_versions=API), "Role")}
    assert not [r for r in roles["r-data-lab-semantic-manager"] if "leases" in r["resources"]]
    assert roles["r-data-lab-semantic-server"] == [{"apiGroups": [""], "resources": ["configmaps"], "verbs": ["get", "list", "watch"]}]


def test_endpoints_keys():
    data = configmap(render(ON, api_versions=API), "r-data-lab-endpoints")
    assert data["SEMANTIC_MCP_URL"] == "http://r-data-lab-semantic.ns.svc:8090/mcp"
    assert data["SEMANTIC_REST_URL"] == "http://r-data-lab-semantic.ns.svc:8090/v1"
    assert data["SEMANTIC_DEFAULT_ROLE"] == "analyst"


def test_one_semantic_release_per_namespace_guard_exists():
    src = (REPO / "charts" / "data-lab" / "templates" / "validate.yaml").read_text()
    assert "one semantic-enabled release per namespace (C14)" in src


def test_pss_and_policy():
    import subprocess
    import sys

    out = subprocess.run(["helm", "template", "r", str(REPO / "charts" / "data-lab"), "-n", "ns", "--kube-version", "1.34.0",
                          "--api-versions", API[0], "--set", "semantic.enabled=true"], capture_output=True, text=True, check=True).stdout
    p = subprocess.run([sys.executable, str(REPO / "hack" / "check-chart-render.py"), "r-data-lab", "ns"],
                       input=out, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout


def test_manager_is_single_and_recreated():
    m = workload(render(ON, api_versions=API), "semantic-manager")
    assert m["spec"]["replicas"] == 1 and m["spec"]["strategy"] == {"type": "Recreate"}


def test_auth_mode_in_endpoints():
    assert configmap(render(ON, api_versions=API), "r-data-lab-endpoints")["SEMANTIC_AUTH_MODE"] == "header"
