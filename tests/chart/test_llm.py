"""The `llm` values block: where the notebooks' AI parts get their models (ollama | openai | none)."""
import json

import pytest
from conftest import RenderError, by_kind, configmap, env_of, render

ENDPOINTS = "r-data-lab-endpoints"
OPENAI = {"provider": "openai", "baseUrl": "https://llm.example.com/v1", "chatModel": "a", "toolModel": "b",
          "auth": {"existingSecret": "k"}}


def jupyter(values):
    return render({"jupyter": {"enabled": True}, **values})


def cfg(docs):
    return configmap(docs, ENDPOINTS)


def test_auto_is_none_without_ollama():
    docs = jupyter({})
    assert cfg(docs)["LLM_PROVIDER"] == "none"
    assert "LLM_BASE_URL" not in cfg(docs)
    assert "LLM_API_KEY" not in env_of(docs, "jupyter")


def test_auto_is_ollama_with_ollama():
    c = cfg(jupyter({"ollama": {"enabled": True}}))
    assert c["LLM_PROVIDER"] == "ollama"
    assert c["LLM_BASE_URL"] == "http://r-data-lab-ollama.ns.svc:11434/v1"
    assert c["LLM_CHAT_MODEL"] == "gemma3:4b"
    assert c["LLM_TOOL_MODEL"] == "qwen3:4b-instruct"
    assert c["LLM_EMBED_MODEL"] == "mxbai-embed-large:latest"


def test_openai_mode():
    docs = jupyter({"llm": OPENAI})
    c, env = cfg(docs), env_of(docs, "jupyter")
    assert c["LLM_PROVIDER"] == "openai" and c["LLM_BASE_URL"] == "https://llm.example.com/v1"
    assert c["LLM_CHAT_MODEL"] == "a" and c["LLM_TOOL_MODEL"] == "b"
    assert env["LLM_API_KEY"]["valueFrom"]["secretKeyRef"] == {"name": "k", "key": "api-key"}
    assert "LLM_EMBED_MODEL" not in c      # set llm.embeddingModel to enable rag_demo's vector sections
    assert "LLM_EMBED_BASE_URL" not in c and "LLM_EMBED_API_KEY" not in env


def test_embeddings_can_use_another_endpoint_and_key():
    docs = jupyter({"llm": {**OPENAI, "embeddingModel": "e", "embeddingBaseUrl": "https://emb.example/v1",
                            "embeddingAuth": {"existingSecret": "k2", "apiKeyKey": "tok"}}})
    c, env = cfg(docs), env_of(docs, "jupyter")
    assert c["LLM_EMBED_BASE_URL"] == "https://emb.example/v1" and c["LLM_EMBED_MODEL"] == "e"
    assert env["LLM_EMBED_API_KEY"]["valueFrom"]["secretKeyRef"] == {"name": "k2", "key": "tok"}
    assert env["LLM_API_KEY"]["valueFrom"]["secretKeyRef"]["name"] == "k"


def test_embeddings_on_the_chart_ollama_while_chat_is_remote():
    docs = jupyter({"ollama": {"enabled": True}, "llm": {**OPENAI, "embeddingModel": "mxbai-embed-large:latest",
                    "embeddingBaseUrl": "http://r-data-lab-ollama.ns.svc:11434/v1"}})
    c, env = cfg(docs), env_of(docs, "jupyter")
    assert c["LLM_PROVIDER"] == "openai" and c["LLM_EMBED_BASE_URL"].endswith(":11434/v1")
    assert "LLM_EMBED_API_KEY" not in env


def test_openai_requires_secret():
    with pytest.raises(RenderError, match=r"llm\.auth\.existingSecret"):
        render({"llm": {**OPENAI, "auth": {"existingSecret": ""}}})


@pytest.mark.parametrize("missing", ["baseUrl", "chatModel", "toolModel"])
def test_openai_requires_models(missing):
    with pytest.raises(RenderError, match=missing):
        render({"llm": {**OPENAI, missing: ""}})


def test_ollama_mode_requires_ollama_enabled():
    with pytest.raises(RenderError, match="ollama.enabled"):
        render({"llm": {"provider": "ollama"}})


def test_unknown_provider_fails():
    with pytest.raises(RenderError, match="llm.provider"):
        render({"llm": {"provider": "azure"}})


def test_embedding_overrides_need_their_prerequisites():
    with pytest.raises(RenderError, match="embeddingModel"):
        render({"llm": {**OPENAI, "embeddingBaseUrl": "https://e/v1"}})
    with pytest.raises(RenderError, match="embeddingBaseUrl"):
        render({"llm": {**OPENAI, "embeddingModel": "e", "embeddingAuth": {"existingSecret": "k2"}}})


def test_ollama_context_length():
    docs = render({"ollama": {"enabled": True}})
    ollama = next(d for d in docs if d["kind"] == "StatefulSet" or d["kind"] == "Deployment"
                  if d["metadata"]["labels"].get("app.kubernetes.io/component") == "ollama")
    env = {e["name"]: e["value"] for e in ollama["spec"]["template"]["spec"]["containers"][0]["env"]}
    assert env["OLLAMA_CONTEXT_LENGTH"] == "16384"


def test_no_key_in_configmaps():
    docs = jupyter({"llm": OPENAI})
    for cm in by_kind(docs, "ConfigMap"):
        for k, v in cm.get("data", {}).items():
            assert not k.startswith("LLM_") or k.endswith(("_URL", "_MODEL", "_PROVIDER")), (cm["metadata"]["name"], k)
            assert "sk-" not in str(v)      # only Secret and key *names* (ENDPOINTS_JSON) appear, never a key


def test_endpoints_json_describes_the_llm():
    llm = json.loads(cfg(jupyter({"llm": OPENAI}))["ENDPOINTS_JSON"])["llm"]
    assert llm["provider"] == "openai" and llm["credentials"] == {"secret": "k", "apiKeyKey": "api-key"}
