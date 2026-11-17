"""Agents on AWS Strands over the lab's MCP servers (not named strands*: that would shadow the package).

* semantic_agent(role)   chat with your data: the Kubeflow Semantic Operator's MCP tools (semantic/01). The only agent
                         that sends traces to Phoenix: call trace_to_phoenix() first (project `semantic-layer`).
* ops_agent(servers)     on-call data engineer: read-only Airflow and Spark History MCP tools (ops/01, ops/02).
* ask(agent, question)   one independent question -> {answer, steps, sql, rows}.

The model is the lab's tool model on its OpenAI-compatible endpoint (lab_utils: llm.toolModel). Guards learned in real
runs with small models are Strands hooks: the model-name guard and the grain guard, plus a cap on tool calls.
"""
import contextlib
import datetime
import json
import os
import re

from strands import Agent
from strands.hooks import BeforeToolCallEvent
from strands.models.openai import OpenAIModel
from strands.tools.mcp import MCPClient

import lab_utils as lu

MAX_TOOL_CALLS = 12          # per question: a small model that keeps calling tools is stopped, not left to loop

SEMANTIC_PROMPT = ("You answer business questions with the semantic layer's tools only. Find certified metrics with "
                   "list_metrics and group-by dimensions with list_dimensions, then call query_metric. Pass only the "
                   "arguments the question needs: `grain` only together with a time dimension, no `limit` unless "
                   "asked. If a tool returns an error, fix the arguments and call it again. Never invent numbers and "
                   "never write SQL. Reply in one or two sentences with the numbers from the result.")

OPS_PROMPT = ("You are the on-call data engineer for Northwind Gear's lakehouse. Answer with the tools: the Airflow "
              "tools for DAGs, runs and task logs; the Spark History tools for Spark applications, jobs and stages. "
              "Never guess ids: list first, then drill down. Do not add filters (dates, states, limits) the question "
              "does not ask for. The Spark History tools take server=\"live\" for the running Spark Connect session and "
              "server=\"history\" for finished applications. Reply in at most five sentences with the facts you found.")


def strands_model() -> OpenAIModel:
    """The lab's tool model through Strands' OpenAI provider (a hosted endpoint, or Ollama at /v1)."""
    lu.require_llm()
    return OpenAIModel(client_args={"base_url": lu.endpoint("LLM_BASE_URL"), "api_key": os.environ.get("LLM_API_KEY") or "unused"},
                       model_id=lu.tool_model(), params={"max_tokens": lu.MAX_TOKENS, "temperature": 0})


# --- tracing (semantic/01 only) -------------------------------------------------------------------------------------
def trace_to_phoenix(project: str = "semantic-layer") -> bool:
    """Send Strands' OpenTelemetry spans to Phoenix over OTLP/HTTP, into `project`. Never raises: when tracing is off it
    prints why. Returns whether it is on."""
    try:
        endpoint = os.environ.get("PHOENIX_OTLP_HTTP_ENDPOINT")
        if not endpoint:
            print("tracing off: Phoenix is not deployed (PHOENIX_OTLP_HTTP_ENDPOINT is not set)")
            return False
        key = os.environ.get("PHOENIX_API_KEY")
        if os.environ.get("PHOENIX_CREDENTIALS_SECRET") and not key:
            print("tracing off: Phoenix authentication is on; set PHOENIX_API_KEY (a Phoenix API key) and re-run")
            return False
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(resource=Resource.create({"service.name": "data-lab-agents",
                                                             "openinference.project.name": project}))
        try:    # rewrites Strands' gen_ai.* spans into OpenInference LLM/tool/agent spans, which Phoenix renders
            from openinference.instrumentation.strands_agents import StrandsAgentsToOpenInferenceProcessor
            provider.add_span_processor(StrandsAgentsToOpenInferenceProcessor())
        except ImportError:
            print("note: openinference-instrumentation-strands-agents is missing; spans arrive as plain gen_ai spans")
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(
            endpoint=endpoint, headers={"authorization": f"Bearer {key}"} if key else None)))
        trace.set_tracer_provider(provider)
        print(f"tracing on: Strands spans go to Phoenix, project {project!r}")
        return True
    except Exception as e:      # tracing must never fail a cell
        print(f"tracing off: {type(e).__name__}: {e}")
        return False


# --- tools and guards ------------------------------------------------------------------------------------------------
def _tools(client, only=None) -> list:
    """The MCP client's tools (optionally only the allowlisted names), with schemas the model server can compile."""
    tools = [t for t in client.list_tools_sync() if only is None or t.tool_name in only]
    for t in tools:
        spec = getattr(t, "mcp_tool", None)
        if spec is not None:
            field = "input_schema" if hasattr(spec, "input_schema") else "inputSchema"
            setattr(spec, field, lu._tool_schema(getattr(spec, field) or {"type": "object"}))
    return tools


class _Guards:
    """Strands hook: fixes a small model's habitual mistakes before a tool runs, and records what it changed."""

    def __init__(self, published=(), time_dims=()):
        self.published, self.time_dims = list(published), set(time_dims)
        self.calls, self.notes = 0, {}

    def register_hooks(self, registry, **kwargs) -> None:
        registry.add_callback(BeforeToolCallEvent, self.before_tool)

    def before_tool(self, event) -> None:
        self.calls += 1
        if self.calls > MAX_TOOL_CALLS:
            event.cancel_tool = f"stopped after {MAX_TOOL_CALLS} tool calls: answer with what you have"
            return
        args, notes = dict(event.tool_use.get("input") or {}), []
        # With exactly one published model an unknown name in a tool call is replaced by it (a real run sent "sales").
        if len(self.published) == 1 and args.get("model") not in (None, self.published[0]):
            notes.append(f"model {args['model']!r} replaced by {self.published[0]!r}")
            args["model"] = self.published[0]
        if event.tool_use["name"] == "query_metric" and args.get("grain") and self.time_dims and \
                not self.time_dims & set(args.get("dimensions") or []):
            notes.append(f"grain dropped: {args.pop('grain')!r} needs a time dimension ({', '.join(sorted(self.time_dims))})")
        event.tool_use["input"] = args
        if notes:
            self.notes[event.tool_use["toolUseId"]] = "; ".join(notes)


def _agent(tools, prompt, guards) -> Agent:
    agent = Agent(model=strands_model(), tools=tools, system_prompt=prompt, hooks=[guards], callback_handler=None)
    agent.lab_guards = guards
    return agent


@contextlib.contextmanager
def semantic_agent(role: str = "analyst"):
    """Chat with your data: an agent over the semantic layer's MCP tools, acting as `role` (row-level governance)."""
    lu.require_semantic()
    lu.require_llm()
    # Name the published model up front: a small model otherwise guesses the optional `model` argument.
    published = [m["name"] for m in lu.mcp_call("list_models", {}, role).get("models", [])]
    time_dims = set()           # the server's contract: `grain` needs one of these among the requested dimensions
    if len(published) == 1:
        dims = lu.mcp_call("list_dimensions", {"model": published[0]}, role).get("dimensions", [])
        time_dims = {d["name"] for d in dims if d.get("isTime")}
    prompt = SEMANTIC_PROMPT
    if published:
        prompt += f" Published semantic model(s): {', '.join(published)}; pass that name as `model` to every tool."
    client = MCPClient(url=os.environ["SEMANTIC_MCP_URL"],
                       headers={"X-Semantic-User": lu.SEMANTIC_USER, "X-Semantic-Role": role})
    with client:
        yield _agent(_tools(client), prompt, _Guards(published, time_dims))


@contextlib.contextmanager
def ops_agent(servers=("airflow", "spark_history")):
    """On-call data engineer: the read-only tools (lab_utils.OPS_TOOLS) of the Airflow and Spark History MCP servers."""
    lu.require_ops()
    lu.require_llm()
    urls = lu.ops_servers()
    clients = [(s, MCPClient(url=urls[s])) for s in servers if s in urls]
    # The model does not know today's date: asked what ran "recently", it filtered on a training-era date.
    today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    with contextlib.ExitStack() as stack:
        tools = []
        for server, client in clients:
            stack.enter_context(client)
            tools += _tools(client, only=lu.OPS_TOOLS.get(server, ()))
        yield _agent(tools, f"{OPS_PROMPT} Today is {today} (UTC).", _Guards())


# --- asking ---------------------------------------------------------------------------------------------------------
def _parse(content) -> object:
    """A Strands toolResult content list -> parsed JSON when the tool returned JSON, else the text."""
    parts = [c["json"] for c in content if "json" in c]
    if parts:
        return parts[0]
    text = "".join(c.get("text", "") for c in content)
    try:
        return json.loads(text)
    except ValueError:
        return text


def _steps(messages, notes) -> list:
    steps, by_id = [], {}
    for msg in messages:
        for block in msg.get("content", []):
            if "toolUse" in block:
                use = block["toolUse"]
                step = {"tool": use["name"], "args": use.get("input") or {}}
                if use["toolUseId"] in notes:
                    step["note"] = notes[use["toolUseId"]]
                by_id[use["toolUseId"]] = step
                steps.append(step)
            elif "toolResult" in block:
                res = block["toolResult"]
                if res["toolUseId"] in by_id:
                    by_id[res["toolUseId"]]["result"] = _parse(res.get("content", []))
    return steps


def ask(agent: Agent, question: str) -> dict:
    """One independent question (the agent forgets earlier ones). Returns {"answer", "steps": [{"tool", "args",
    "result", "note"?}], "sql", "rows"}; sql and rows are those of the last query_metric."""
    agent.messages.clear()
    guards = agent.lab_guards
    guards.calls, guards.notes = 0, {}
    answer = str(agent(question)).strip()
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.S).strip()      # thinking models may reason inline
    steps = _steps(agent.messages, guards.notes)
    sql, rows = None, []
    for step in steps:
        out = step.get("result")
        if step["tool"] == "query_metric" and isinstance(out, dict) and "sql" in out:
            sql, rows = out["sql"], out.get("rows", [])
    return {"answer": answer, "steps": steps, "sql": sql, "rows": rows}
