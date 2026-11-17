"""lab_utils talks to every model through the OpenAI-compatible API (a hosted endpoint, or Ollama at /v1)."""
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "jupyter" / "notebooks"))
import lab_utils  # noqa: E402


def completion(content="ok", tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def tool_call(name, arguments):
    return SimpleNamespace(id="c1", function=SimpleNamespace(name=name, arguments=arguments))


class FakeClient:
    def __init__(self, reply=None, error=None):
        self.kwargs, self.reply, self.error = None, reply or completion(), error
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.reply


def status_error(cls, code, text="boom"):
    req = httpx.Request("POST", "https://x/v1/chat/completions")
    return cls(text, response=httpx.Response(code, request=req), body=None)


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example/v1")
    monkeypatch.setenv("LLM_API_KEY", "secret")
    for k in ("LLM_EMBED_MODEL", "LLM_EMBED_BASE_URL", "LLM_EMBED_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_chat_decodes_tool_call_arguments(env):
    fake = FakeClient(completion(None, [tool_call("list_models", '{"model":"dbt_demo"}')]))
    env.setattr(lab_utils, "openai_client", lambda: fake)
    msg = lab_utils._chat("m", [{"role": "user", "content": "hi"}])["message"]
    assert msg["tool_calls"][0]["function"] == {"name": "list_models", "arguments": {"model": "dbt_demo"}}
    assert msg["content"] == ""


def test_chat_caps_max_tokens(env):
    fake = FakeClient()
    env.setattr(lab_utils, "openai_client", lambda: fake)
    lab_utils._chat("m", [{"role": "user", "content": "hi"}])
    assert fake.kwargs["max_tokens"] == lab_utils.MAX_TOKENS == 1024 and fake.kwargs["temperature"] == 0
    assert not {"think", "num_ctx", "options"} & set(fake.kwargs)


def test_chat_error_names_endpoint(env):
    env.setattr(lab_utils, "openai_client", lambda: FakeClient(error=status_error(openai.AuthenticationError, 401)))
    with pytest.raises(RuntimeError) as e:
        lab_utils._chat("m", [{"role": "user", "content": "hi"}])
    assert "LLM_BASE_URL" in str(e.value) and "401" in str(e.value) and "https://llm.example/v1" in str(e.value)
    assert "secret" not in str(e.value)


@pytest.mark.parametrize("provider,expected", [("none", False), ("", False), ("ollama", True), ("openai", True)])
def test_llm_available_modes(monkeypatch, provider, expected):
    monkeypatch.setenv("LLM_PROVIDER", provider)
    assert lab_utils.llm_available() is expected


def test_skip_unless_llm_prints(monkeypatch, capsys):
    monkeypatch.setenv("LLM_PROVIDER", "none")
    assert lab_utils.skip_unless_llm("answer in plain English") is False
    assert capsys.readouterr().out.startswith("skipped (no LLM configured): answer in plain English")
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    assert lab_utils.skip_unless_llm("x") is True and capsys.readouterr().out == ""


def test_require_llm_names_the_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "none")
    with pytest.raises(RuntimeError, match="LLM_PROVIDER=none"):
        lab_utils.require_llm()


def test_embeddings_available_false_without_model(env):
    assert lab_utils.embeddings_available() is False
    env.setenv("LLM_EMBED_MODEL", "e")
    assert lab_utils.embeddings_available() is True
    env.setenv("LLM_PROVIDER", "none")
    assert lab_utils.embeddings_available() is False


def test_embeddings_default_to_the_chat_endpoint_and_key(env):
    env.setenv("LLM_EMBED_MODEL", "e")
    assert lab_utils.embeddings_config() == {"base_url": "https://llm.example/v1", "api_key": "secret", "model": "e"}


def test_embeddings_override_never_receives_the_chat_key(env):
    env.setenv("LLM_EMBED_MODEL", "e")
    env.setenv("LLM_EMBED_BASE_URL", "https://other.example/v1")
    assert lab_utils.embeddings_config()["api_key"] == "unused"          # not the chat provider's key
    env.setenv("LLM_EMBED_API_KEY", "k2")
    assert lab_utils.embeddings_config() == {"base_url": "https://other.example/v1", "api_key": "k2", "model": "e"}


def test_embed_returns_vectors_in_order(env):
    env.setenv("LLM_EMBED_MODEL", "e")
    seen = {}

    class Emb:
        def create(self, **kw):
            seen.update(kw)
            return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 2.0]), SimpleNamespace(embedding=[3.0])])

    env.setattr(lab_utils, "embeddings_client", lambda: SimpleNamespace(embeddings=Emb()))
    assert lab_utils.embed(["a", "b"]) == [[1.0, 2.0], [3.0]] and seen == {"model": "e", "input": ["a", "b"]}


def test_require_tool_model_probe_rejects_no_tools(env):
    err = status_error(openai.BadRequestError, 400, "tools not supported")
    env.setattr(lab_utils, "openai_client", lambda: FakeClient(error=err))
    with pytest.raises(RuntimeError, match="chat-only-model"):
        lab_utils.require_tool_model("chat-only-model")


def test_require_tool_model_probe_accepts_tools(env):
    fake = FakeClient()
    env.setattr(lab_utils, "openai_client", lambda: fake)
    lab_utils.require_tool_model("good")
    assert fake.kwargs["model"] == "good" and len(fake.kwargs["tools"]) == 1


def test_tool_model_defaults_only_for_ollama(monkeypatch):
    monkeypatch.delenv("LLM_TOOL_MODEL", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    assert lab_utils.tool_model() == "qwen3:4b-instruct"
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert lab_utils.tool_model() == ""
    monkeypatch.setenv("LLM_TOOL_MODEL", "m")
    assert lab_utils.tool_model() == "m"
