"""Ollama GPU: device plugin (extended resource) or DRA (namespaced ResourceClaimTemplate)."""
import pytest

from conftest import RenderError, by_kind, render, workload

DRA_API = ["resource.k8s.io/v1/ResourceClaimTemplate"]


def ollama(gpu):
    return {"ollama": {"enabled": True, "gpu": gpu}}


def container(docs):
    return workload(docs, "ollama")["spec"]["template"]["spec"]["containers"][0]


def test_device_plugin_limit():
    docs = render(ollama({"enabled": True}))
    assert container(docs)["resources"]["limits"]["nvidia.com/gpu"] == 1
    assert not by_kind(docs, "ResourceClaimTemplate")


def test_dra_claim():
    docs = render(ollama({"enabled": True, "dra": {"enabled": True}}), api_versions=DRA_API)
    [rct] = by_kind(docs, "ResourceClaimTemplate")
    assert rct["metadata"]["name"] == "r-data-lab-ollama-gpu"
    assert rct["apiVersion"] == "resource.k8s.io/v1"
    assert rct["spec"]["spec"]["devices"]["requests"][0]["exactly"]["deviceClassName"] == "gpu.nvidia.com"
    pod = workload(docs, "ollama")["spec"]["template"]["spec"]
    assert pod["resourceClaims"] == [{"name": "gpu", "resourceClaimTemplateName": "r-data-lab-ollama-gpu"}]
    c = container(docs)
    assert c["resources"]["claims"] == [{"name": "gpu"}]
    assert "nvidia.com/gpu" not in c["resources"].get("limits", {})


def test_dra_requires_api():
    with pytest.raises(RenderError, match="resource.k8s.io/v1"):
        render(ollama({"enabled": True, "dra": {"enabled": True}}))


def test_dra_requires_gpu_enabled():
    with pytest.raises(RenderError, match="ollama.gpu.enabled"):
        render(ollama({"enabled": False, "dra": {"enabled": True}}), api_versions=DRA_API)
