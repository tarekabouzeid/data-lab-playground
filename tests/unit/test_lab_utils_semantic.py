"""Chat-with-your-data helpers in lab_utils.py (no network: the LLM, MCP and Trino are faked)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "jupyter" / "notebooks"))
import lab_utils  # noqa: E402

QUERY_RESULT = {"columns": ["total_revenue"], "rows": [[40.0]], "rowCount": 1,
                "sql": "SELECT SUM(CASE WHEN status <> 'returned' THEN amount ELSE 0 END) FROM fct_orders"}


def test_mcp_schema_to_ollama():
    tool = {"name": "query_metric", "description": "d", "inputSchema": {"type": "object", "properties": {"metrics": {"type": "array"}}}}
    assert lab_utils.mcp_tool_to_ollama(tool) == {
        "type": "function",
        "function": {"name": "query_metric", "description": "d",
                     "parameters": {"type": "object", "properties": {"metrics": {"type": "array"}}}},
    }


def test_require_semantic_message(monkeypatch):
    monkeypatch.delenv("SEMANTIC_MCP_URL", raising=False)
    assert not lab_utils.semantic_available()
    with pytest.raises(RuntimeError) as e:
        lab_utils.require_semantic()
    assert str(e.value) == ("The semantic layer runs only on the Helm chart with semantic.enabled=true "
                            "(Docker Compose is frozen and does not include it).")


def test_wait_for_model_timeout(monkeypatch):
    monkeypatch.setattr(lab_utils, "mcp_call", lambda name, args, role="analyst": {"models": [{"name": "other"}]})
    monkeypatch.setattr(lab_utils.time, "sleep", lambda s: None)
    with pytest.raises(TimeoutError, match="dbt_demo"):
        lab_utils.wait_for_model("dbt_demo", timeout_s=0)


def test_wait_for_model_found(monkeypatch):
    monkeypatch.setattr(lab_utils, "mcp_call", lambda name, args, role="analyst": {"models": [{"name": "dbt_demo"}]})
    lab_utils.wait_for_model("dbt_demo", timeout_s=1)


class Frame:
    """The two pandas.DataFrame accessors ask_raw_sql uses (pandas is not needed to run these tests)."""

    def __init__(self, rows):
        self.rows = rows
        self.iloc = {(0, 0): rows[0][0]}

    @property
    def values(self):
        return type("V", (), {"tolist": lambda _self: self.rows})()


def fake_chat(script):
    """A fake Ollama chat that answers from a list of messages, recording what it was sent."""
    sent = []

    def chat(model, messages, tools=None):
        sent.append(list(messages))
        return {"message": script[min(len(sent), len(script)) - 1]}
    chat.sent = sent
    return chat


def call(name, args):
    return {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]}


def test_ask_semantic_loop(monkeypatch):
    chat = fake_chat([call("list_metrics", {}), call("query_metric", {"metrics": ["total_revenue"]}),
                      {"role": "assistant", "content": "Total revenue is 40."}])
    calls = []

    def mcp(name, args, role="analyst"):
        calls.append((name, args, role))
        return QUERY_RESULT if name == "query_metric" else {"metrics": [{"name": "total_revenue"}]}
    monkeypatch.setattr(lab_utils, "_chat", chat)
    monkeypatch.setattr(lab_utils, "mcp_tools", lambda role="analyst": [])
    monkeypatch.setattr(lab_utils, "mcp_call", mcp)
    out = lab_utils.ask_semantic("What is total revenue?", role="marketing")
    assert out["tool_calls"] == ["list_metrics", "query_metric"]
    assert out["sql"] == QUERY_RESULT["sql"]
    assert out["rows"] == [[40.0]]
    assert out["answer"] == "Total revenue is 40."
    assert {r for _, _, r in calls} == {"marketing"}
    assert chat.sent[-1][-1]["role"] == "tool"


def test_ask_semantic_turn_cap(monkeypatch):
    monkeypatch.setattr(lab_utils, "_chat", fake_chat([call("list_metrics", {})]))
    monkeypatch.setattr(lab_utils, "mcp_tools", lambda role="analyst": [])
    monkeypatch.setattr(lab_utils, "mcp_call", lambda name, args, role="analyst": {"metrics": []})
    out = lab_utils.ask_semantic("loop forever", max_turns=3)
    assert out["tool_calls"] == ["list_metrics"] * 3
    assert "3 turns" in out["answer"]


def test_ask_raw_sql_runs_the_generated_sql(monkeypatch):
    monkeypatch.setattr(lab_utils, "_chat", fake_chat([{"role": "assistant", "content": "```sql\nSELECT sum(amount) FROM fct_orders\n```"}]))
    seen = []

    def trino(sql):
        seen.append(sql)
        return Frame([[50.0]]) if "sum(amount)" in sql else Frame([["fct_orders DDL"]])
    monkeypatch.setattr(lab_utils, "trino", trino)
    out = lab_utils.ask_raw_sql("total revenue?", tables=["fct_orders"])
    assert out["sql"] == "SELECT sum(amount) FROM fct_orders"
    assert out["rows"] == [[50.0]] and out["error"] is None


def test_ask_raw_sql_reports_sql_errors(monkeypatch):
    monkeypatch.setattr(lab_utils, "_chat", fake_chat([{"role": "assistant", "content": "SELECT nope FROM fct_orders"}]))

    def trino(sql):
        if "nope" in sql:
            raise RuntimeError("Column 'nope' cannot be resolved")
        return Frame([["ddl"]])
    monkeypatch.setattr(lab_utils, "trino", trino)
    out = lab_utils.ask_raw_sql("?", tables=["fct_orders"])
    assert out["rows"] == [] and "cannot be resolved" in out["error"]


def test_require_tool_model(monkeypatch):
    monkeypatch.setattr(lab_utils, "_ollama_show", lambda model: {"capabilities": ["completion"]})
    with pytest.raises(RuntimeError, match="gemma3:4b"):
        lab_utils.require_tool_model("gemma3:4b")
    monkeypatch.setattr(lab_utils, "_ollama_show", lambda model: {"capabilities": ["completion", "tools"]})
    lab_utils.require_tool_model("qwen3:4b")


def test_mcp_call_parses_tool_result(monkeypatch):
    replies = [{"result": {}}, None, {"result": {"content": [{"type": "text", "text": json.dumps(QUERY_RESULT)}]}}]
    sent = []

    def rpc(url, headers, body):
        sent.append((headers, body))
        return replies[len(sent) - 1]
    monkeypatch.setenv("SEMANTIC_MCP_URL", "http://semantic:8090/mcp")
    monkeypatch.setattr(lab_utils, "_mcp_post", rpc)
    assert lab_utils.mcp_call("query_metric", {"metrics": ["total_revenue"]}, role="analyst") == QUERY_RESULT
    assert sent[0][0]["X-Semantic-Role"] == "analyst" and sent[0][0]["X-Semantic-User"] == "lab"


def test_mcp_call_tool_error(monkeypatch):
    replies = [{"result": {}}, None, {"result": {"isError": True, "content": [{"type": "text", "text": "unauthorized: role"}]}}]
    monkeypatch.setenv("SEMANTIC_MCP_URL", "http://semantic:8090/mcp")
    monkeypatch.setattr(lab_utils, "_mcp_post", lambda url, headers, body: replies.pop(0))
    assert lab_utils.mcp_call("query_metric", {}) == {"error": "unauthorized: role"}
