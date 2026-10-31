"""Optional Gateway API routes: off by default, namespaced only, hostnames unique per release."""
import pytest

from conftest import RenderError, by_kind, configmap, render

API = ["gateway.networking.k8s.io/v1/Gateway"]
GW = {"gateway": {"enabled": True, "gatewayClassName": "datalab"}, "jupyter": {"enabled": True}}


def hosts(docs):
    return {r["metadata"]["name"]: r["spec"]["hostnames"][0] for r in by_kind(docs, "HTTPRoute")}


def test_off_by_default():
    docs = render(api_versions=API)
    assert not by_kind(docs, "Gateway") and not by_kind(docs, "HTTPRoute")


def test_namespaced_gateway():
    docs = render(GW, api_versions=API)
    assert [g["metadata"]["name"] for g in by_kind(docs, "Gateway")] == ["r-data-lab-gateway"]
    assert by_kind(docs, "Gateway")[0]["spec"]["gatewayClassName"] == "datalab"
    assert not by_kind(docs, "GatewayClass")
    for r in by_kind(docs, "HTTPRoute"):
        assert r["spec"]["parentRefs"] == [{"name": "r-data-lab-gateway"}]


def test_hostnames_include_release():
    h = hosts(render(GW, release="datalab", api_versions=API))
    assert h["datalab-data-lab-jupyter"] == "jupyter.datalab.localhost"
    a = set(hosts(render(GW, release="a", api_versions=API)).values())
    b = set(hosts(render(GW, release="b", api_versions=API)).values())
    assert a and not a & b


def test_routes_follow_enabled_components():
    h = hosts(render(GW, api_versions=API))
    assert {"r-data-lab-jupyter", "r-data-lab-trino", "r-data-lab-spark", "r-data-lab-airflow"} <= set(h)
    assert "r-data-lab-ollama" not in h  # ollama is off


def test_longlived_jupyter():
    route = next(r for r in by_kind(render(GW, api_versions=API), "HTTPRoute") if r["metadata"]["name"] == "r-data-lab-jupyter")
    assert route["spec"]["rules"][0]["timeouts"] == {"request": "0s"}
    assert route["spec"]["rules"][0]["backendRefs"] == [{"name": "r-data-lab-jupyter", "port": 8888}]


def test_extra_routes():
    v = {**GW, "gateway": {**GW["gateway"], "extraRoutes": [{"name": "headlamp", "service": "headlamp", "port": 80, "longLived": True}]}}
    route = next(r for r in by_kind(render(v, api_versions=API), "HTTPRoute") if r["metadata"]["name"] == "r-data-lab-headlamp")
    assert route["spec"]["rules"][0]["backendRefs"] == [{"name": "headlamp", "port": 80}]


def test_shared_parent_ref():
    v = {"gateway": {"enabled": True, "parentRef": {"name": "shared", "namespace": "infra", "sectionName": "http"}}}
    docs = render(v, api_versions=API)
    assert not by_kind(docs, "Gateway")
    assert by_kind(docs, "HTTPRoute")[0]["spec"]["parentRefs"] == [{"name": "shared", "namespace": "infra", "sectionName": "http"}]


def test_parentref_xor_class():
    with pytest.raises(RenderError, match="gatewayClassName or gateway.parentRef"):
        render({"gateway": {"enabled": True, "gatewayClassName": "x", "parentRef": {"name": "y"}}}, api_versions=API)
    with pytest.raises(RenderError, match="gatewayClassName or gateway.parentRef"):
        render({"gateway": {"enabled": True}}, api_versions=API)


def test_requires_gateway_api():
    with pytest.raises(RenderError, match="Gateway API"):
        render(GW)


def test_netpol_allows_gateway_ns():
    v = {**GW, "gateway": {**GW["gateway"], "allowFromNamespaces": ["envoy-gateway-system"]}}
    peers = by_kind(render(v, api_versions=API), "NetworkPolicy")[0]["spec"]["ingress"][0]["from"]
    assert {"namespaceSelector": {"matchLabels": {"kubernetes.io/metadata.name": "envoy-gateway-system"}}} in peers


def test_trino_forwarded():
    def props(docs):
        return configmap(docs, "r-data-lab-trino")["config.properties"]
    assert "http-server.process-forwarded=true" in props(render(GW, api_versions=API))
    assert "process-forwarded" not in props(render())


def test_gateway_urls_in_endpoints():
    data = configmap(render(GW, release="datalab", api_versions=API), "datalab-data-lab-endpoints")
    assert data["GATEWAY_URL_JUPYTER"] == "http://jupyter.datalab.localhost"
