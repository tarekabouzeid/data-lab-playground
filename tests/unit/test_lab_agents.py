"""lab_agents: Strands agents over the MCP servers. The model is scripted and the MCP tools are fakes."""
import json
import sys
from pathlib import Path

import pytest
from strands import tool
from strands.models import Model

NB = Path(__file__).resolve().parents[2] / "jupyter" / "notebooks"
sys.path.insert(0, str(NB))
import lab_agents  # noqa: E402
import lab_utils  # noqa: E402

QUERY_RESULT = {"columns": ["total_revenue"], "rows": [[40.0]], "rowCount": 1, "sql": "SELECT 40"}


class ScriptedModel(Model):
    """Answers from a script: a ("tool", name, input) step is a tool use, a ("text", s) step the final answer."""

    def __init__(self, script):
        self.script, self.i, self.systems = script, 0, []

    def update_config(self, **kw): ...
    def get_config(self): return {}
    async def structured_output(self, *a, **kw):
        raise NotImplementedError
        yield

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kw):
        self.systems.append(system_prompt)
        step = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        yield {"messageStart": {"role": "assistant"}}
        if step[0] == "tool":
            yield {"contentBlockStart": {"start": {"toolUse": {"toolUseId": f"t{self.i}", "name": step[1]}}}}
            yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(step[2])}}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            yield {"contentBlockStart": {"start": {}}}
            yield {"contentBlockDelta": {"delta": {"text": step[1]}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}
        yield {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}, "metrics": {"latencyMs": 1}}}


def make_tool(name, calls, result):
    @tool(name=name)
    def fn(model: str = "", metrics: list = None, dimensions: list = None, grain: str = "") -> str:
        """fake MCP tool"""
        calls.append((name, {"model": model, "metrics": metrics, "dimensions": dimensions, "grain": grain}))
        return json.dumps(result)
    return fn


class FakeMCP:
    instances = []
    tools_by_url = {}

    def __init__(self, url=None, headers=None, **kw):
        self.url, self.headers = url, headers
        FakeMCP.instances.append(self)

    def __enter__(self): return self
    def __exit__(self, *a): return False
    def list_tools_sync(self): return FakeMCP.tools_by_url[self.url]


@pytest.fixture
def stack(monkeypatch):
    FakeMCP.instances, FakeMCP.tools_by_url = [], {}
    monkeypatch.setattr(lab_agents, "MCPClient", FakeMCP)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_BASE_URL", "http://x/v1")
    monkeypatch.setenv("LLM_TOOL_MODEL", "m")
    monkeypatch.setenv("SEMANTIC_MCP_URL", "http://sem/mcp")
    monkeypatch.setenv("AIRFLOW_MCP_URL", "http://af/mcp")
    monkeypatch.setenv("SPARK_HISTORY_MCP_URL", "http://sh/mcp")
    monkeypatch.delenv("SEMANTIC_AUTH_MODE", raising=False)
    calls = []
    FakeMCP.tools_by_url["http://sem/mcp"] = [make_tool("query_metric", calls, QUERY_RESULT), make_tool("list_models", calls, {})]

    def mcp_call(name, args, role="analyst"):
        return {"models": [{"name": "dbt_demo"}]} if name == "list_models" else \
            {"dimensions": [{"name": "fct_orders.order_date", "isTime": True}, {"name": "dim_products.category"}]}
    monkeypatch.setattr(lab_utils, "mcp_call", mcp_call)
    return monkeypatch, calls


def script(monkeypatch, *steps):
    model = ScriptedModel(list(steps))
    monkeypatch.setattr(lab_agents, "strands_model", lambda: model)
    return model


def test_strands_model_uses_llm_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_BASE_URL", "http://x/v1")
    monkeypatch.setenv("LLM_TOOL_MODEL", "m")
    m = lab_agents.strands_model()
    cfg = m.get_config()
    assert cfg["model_id"] == "m" and cfg["params"]["max_tokens"] == 1024


def test_semantic_agent_sends_identity_headers(stack):
    mp, _ = stack
    script(mp, ("text", "ok"))
    with lab_agents.semantic_agent("dach_sales"):
        pass
    assert FakeMCP.instances[0].headers["X-Semantic-Role"] == "dach_sales"
    assert FakeMCP.instances[0].headers["X-Semantic-User"] == lab_utils.SEMANTIC_USER


def test_semantic_agent_prompt_names_published_model(stack):
    mp, _ = stack
    model = script(mp, ("text", "hi"))
    with lab_agents.semantic_agent() as agent:
        lab_agents.ask(agent, "q")
    assert "dbt_demo" in model.systems[0]


def test_model_guard_hook(stack):
    mp, calls = stack
    script(mp, ("tool", "query_metric", {"model": "sales", "metrics": ["total_revenue"]}), ("text", "40"))
    with lab_agents.semantic_agent() as agent:
        out = lab_agents.ask(agent, "revenue?")
    assert calls[0][1]["model"] == "dbt_demo"
    assert "model" in out["steps"][0]["note"] and out["steps"][0]["args"]["model"] == "dbt_demo"


def test_grain_guard_hook(stack):
    mp, calls = stack
    script(mp, ("tool", "query_metric", {"metrics": ["m"], "grain": "day", "dimensions": ["dim_products.category"]}), ("text", "x"))
    with lab_agents.semantic_agent() as agent:
        out = lab_agents.ask(agent, "q")
    assert calls[0][1]["grain"] == "" and out["steps"][0]["note"].startswith("grain dropped")
    calls.clear()
    script(mp, ("tool", "query_metric", {"metrics": ["m"], "grain": "day", "dimensions": ["fct_orders.order_date"]}), ("text", "x"))
    with lab_agents.semantic_agent() as agent:
        out = lab_agents.ask(agent, "q")
    assert calls[0][1]["grain"] == "day" and "note" not in out["steps"][0]


def test_ask_collects_steps_sql_rows(stack):
    mp, _ = stack
    script(mp, ("tool", "query_metric", {"model": "dbt_demo", "metrics": ["total_revenue"]}), ("text", "Revenue is 40."))
    with lab_agents.semantic_agent() as agent:
        out = lab_agents.ask(agent, "revenue?")
    assert out["answer"] == "Revenue is 40." and out["sql"] == QUERY_RESULT["sql"] and out["rows"] == QUERY_RESULT["rows"]
    assert out["steps"][0]["tool"] == "query_metric" and out["steps"][0]["result"] == QUERY_RESULT


def test_tool_calls_are_capped(stack):
    mp, calls = stack
    script(mp, ("tool", "query_metric", {"metrics": ["m"]}))      # never answers
    with lab_agents.semantic_agent() as agent:
        agent.lab_guards.calls = 0
        try:
            lab_agents.ask(agent, "loop")
        except Exception:
            pass
    assert len(calls) <= lab_agents.MAX_TOOL_CALLS


def test_ops_agent_offers_only_allowlisted_tools(stack):
    mp, calls = stack
    FakeMCP.tools_by_url["http://af/mcp"] = [make_tool(n, calls, {}) for n in ("trigger_dag", "list_dag_runs", "get_task_logs")]
    FakeMCP.tools_by_url["http://sh/mcp"] = [make_tool("delete_everything", calls, {})]
    script(mp, ("text", "ok"))
    with lab_agents.ops_agent() as agent:
        assert set(agent.tool_names) == {"list_dag_runs", "get_task_logs"}


def test_ops_agent_prompt_has_today(stack):
    import datetime
    mp, calls = stack
    FakeMCP.tools_by_url["http://af/mcp"] = []
    FakeMCP.tools_by_url["http://sh/mcp"] = []
    model = script(mp, ("text", "ok"))
    with lab_agents.ops_agent() as agent:
        lab_agents.ask(agent, "what ran?")
    assert datetime.datetime.now(datetime.timezone.utc).date().isoformat() in model.systems[0]


def test_agents_raise_without_llm(stack, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "none")
    with pytest.raises(RuntimeError, match="LLM_PROVIDER=none"):
        with lab_agents.semantic_agent():
            pass
    with pytest.raises(RuntimeError, match="LLM_PROVIDER=none"):
        with lab_agents.ops_agent():
            pass


def test_tracing_never_raises(stack, monkeypatch, capsys):
    monkeypatch.delenv("PHOENIX_OTLP_HTTP_ENDPOINT", raising=False)
    assert lab_agents.trace_to_phoenix() is False and "tracing off" in capsys.readouterr().out
    monkeypatch.setenv("PHOENIX_OTLP_HTTP_ENDPOINT", "http://p:6006/v1/traces")
    monkeypatch.setenv("PHOENIX_CREDENTIALS_SECRET", "s")
    monkeypatch.delenv("PHOENIX_API_KEY", raising=False)
    assert lab_agents.trace_to_phoenix() is False and "PHOENIX_API_KEY" in capsys.readouterr().out
    monkeypatch.delenv("PHOENIX_CREDENTIALS_SECRET")
    import opentelemetry.sdk.trace as sdk
    monkeypatch.setattr(sdk, "TracerProvider", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert lab_agents.trace_to_phoenix() is False
    mp, _ = stack
    script(mp, ("text", "still works"))
    with lab_agents.semantic_agent() as agent:
        assert lab_agents.ask(agent, "q")["answer"] == "still works"


def test_only_semantic_notebook_traces():
    hits = [f.name for f in NB.rglob("*.ipynb") if ".ipynb_checkpoints" not in f.parts and "trace_to_phoenix(" in f.read_text()]
    assert hits == ["01_chat_with_your_data.ipynb"]


def test_spark_server_guard_fills_a_missing_server(stack):
    """A real run: qwen3:4b-instruct called get_sql_execution on the live app without `server`, so the MCP server used
    its default backend (history) and answered 404 for every call. ops_agent(spark_server="live") fills it in."""
    mp, calls = stack

    @tool(name="get_sql_execution")
    def get_sql_execution(app_id: str, execution_id: int, server: str = "") -> str:
        """fake Spark History tool"""
        calls.append(("get_sql_execution", {"app_id": app_id, "server": server}))
        return "{}"

    FakeMCP.tools_by_url["http://sh/mcp"] = [get_sql_execution]
    script(mp, ("tool", "get_sql_execution", {"app_id": "spark-1", "execution_id": 3}),
           ("tool", "get_sql_execution", {"app_id": "spark-0", "execution_id": 1, "server": "history"}), ("text", "x"))
    with lab_agents.ops_agent(("spark_history",), spark_server="live") as agent:
        out = lab_agents.ask(agent, "q")
    assert [c[1]["server"] for c in calls] == ["live", "history"]       # an explicit server is never overridden
    assert "server" in out["steps"][0]["note"] and "note" not in out["steps"][1]
    calls.clear()
    script(mp, ("tool", "get_sql_execution", {"app_id": "spark-1", "execution_id": 3}), ("text", "x"))
    with lab_agents.ops_agent(("spark_history",)) as agent:              # default: no guard, the server decides
        lab_agents.ask(agent, "q")
    assert calls[0][1]["server"] == ""


def test_tool_results_are_capped(stack):
    """A real run against a strict 20k-context endpoint: ops/02's get_sql_execution results grew one question to 35k
    tokens. (Ollama silently drops old context instead.) Each tool result reaches the model as at most
    MAX_TOOL_RESULT_CHARS characters, and the step says it was cut."""
    mp, calls = stack

    @tool(name="get_sql_execution")
    def get_sql_execution(app_id: str = "") -> str:
        """fake Spark History tool with a huge result"""
        return "x" * 50_000

    FakeMCP.tools_by_url["http://sh/mcp"] = [get_sql_execution]
    script(mp, ("tool", "get_sql_execution", {"app_id": "a"}), ("text", "done"))
    with lab_agents.ops_agent(("spark_history",)) as agent:
        out = lab_agents.ask(agent, "q")
        sent = [c["text"] for m in agent.messages for b in m["content"] if "toolResult" in b
                for c in b["toolResult"]["content"] if "text" in c]
    assert sent and all(len(t) <= lab_agents.MAX_TOOL_RESULT_CHARS + 100 for t in sent)
    assert "cut" in out["steps"][0]["note"]


def test_litellm_context_overflow_lets_strands_recover(monkeypatch):
    """LiteLLM / llama.cpp say "exceeds the available context size", which Strands 1.65 does not classify as a context
    overflow, so its conversation manager never truncated old tool results and the request just failed (ops/02 on a
    20k-context gateway). The lab's model raises ContextWindowOverflowException for it."""
    import asyncio
    import httpx
    import openai
    from strands.models.openai import OpenAIModel
    from strands.types.exceptions import ContextWindowOverflowException

    def raising(message):
        async def stream(self, *a, **kw):
            raise openai.BadRequestError(message, response=httpx.Response(400, request=httpx.Request("POST", "http://x")), body=None)
            yield
        return stream

    async def drain(model):
        async for _ in model.stream([], None, None):
            pass

    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_BASE_URL", "http://x/v1")
    monkeypatch.setenv("LLM_TOOL_MODEL", "m")
    model = lab_agents.strands_model()
    monkeypatch.setattr(OpenAIModel, "stream", raising(
        "litellm.ContextWindowExceededError: request (24871 tokens) exceeds the available context size (20480 tokens)"))
    with pytest.raises(ContextWindowOverflowException):
        asyncio.run(drain(model))
    monkeypatch.setattr(OpenAIModel, "stream", raising("model not found"))
    with pytest.raises(openai.BadRequestError):            # other errors pass through unchanged
        asyncio.run(drain(model))
