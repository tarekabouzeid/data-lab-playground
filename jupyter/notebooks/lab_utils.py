"""Helpers shared by the technique notebooks (iceberg/, dbt/).

Runs in the lab's Jupyter (Helm chart charts/data-lab), where Spark is Spark Connect (SPARK_REMOTE is set). The Spark
Connect server already has the catalog, the Iceberg SQL extensions and the S3A settings
(charts/data-lab/templates/spark-configmap.yaml). Do not call .master() here: pyspark 4.1 rejects it together with
SPARK_REMOTE, and static confs cannot be set by a Connect client.

Every service endpoint lives here as an environment variable set by the chart. Notebooks import these constants
instead of hardcoding hosts (tests/unit/test_notebook_endpoints.py). A constant whose variable is missing raises one
clear error (run the notebook in the lab's Jupyter) instead of connecting to a made-up host name.

Imports of optional packages (boto3, trino, pyarrow) are lazy so the module imports anywhere.
"""
import json
import os
import re
import time
from pathlib import Path

CATALOG = "iceberg_catalog"                 # Spark name of the HMS Iceberg REST catalog
TRINO_CATALOG = "iceberg"                   # the same catalog as seen by Trino
TRINO_PORT = int(os.environ.get("TRINO_PORT", "8080"))
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY") or None   # set when the chart enables Qdrant API keys

_ENDPOINTS = {                                  # module constant -> environment variable set by the chart
    "REST_URI": "ICEBERG_REST_URI", "S3_ENDPOINT": "S3_ENDPOINT", "S3_KEY": "S3_ACCESS_KEY", "S3_SECRET": "S3_SECRET_KEY",
    "TRINO_HOST": "TRINO_HOST", "OLLAMA_URL": "OLLAMA_URL", "QDRANT_URL": "QDRANT_URL", "PHOENIX_URL": "PHOENIX_URL",
    "PHOENIX_OTLP_ENDPOINT": "PHOENIX_OTLP_GRPC_ENDPOINT",
}


def endpoint(name: str) -> str:
    """The value of environment variable `name`, or a clear error when the notebook runs outside the lab."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set: run this notebook in the lab's Jupyter (Helm chart charts/data-lab)")
    return value


def __getattr__(const: str):                    # `from lab_utils import TRINO_HOST` and `lab_utils.REST_URI`
    if const in _ENDPOINTS:
        return endpoint(_ENDPOINTS[const])
    if const in ("TOOL_MODEL", "CHAT_MODEL", "EMBED_MODEL"):           # read from the environment on every access
        return {"TOOL_MODEL": tool_model, "CHAT_MODEL": chat_model, "EMBED_MODEL": embed_model}[const]()
    raise AttributeError(f"module {__name__!r} has no attribute {const!r}")


def on_spark_connect() -> bool:
    return bool(os.environ.get("SPARK_REMOTE"))


def get_spark(app_name: str = "lab-notebook"):
    """A Spark Connect session on the lab's Spark (the server has the Iceberg catalog)."""
    from pyspark.sql import SparkSession

    endpoint("SPARK_REMOTE")
    return SparkSession.builder.appName(app_name).getOrCreate()


def trino(sql: str):
    """Run one statement on Trino (catalog `iceberg`) and return a pandas DataFrame."""
    import pandas as pd
    import trino as trino_client

    conn = trino_client.dbapi.connect(host=endpoint("TRINO_HOST"), port=TRINO_PORT, user="lab", catalog=TRINO_CATALOG)
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    cols = [c.name for c in (cur.description or [])]
    return pd.DataFrame(rows, columns=cols)


def s3_client():
    import boto3

    return boto3.client("s3", endpoint_url=endpoint("S3_ENDPOINT"), aws_access_key_id=endpoint("S3_ACCESS_KEY"),
                        aws_secret_access_key=endpoint("S3_SECRET_KEY"), region_name="us-east-1")


def parquet_footer(s3a_path: str):
    """Parquet footer metadata of an s3a:// object (downloads the file; use small files)."""
    import io

    import pyarrow.parquet as pq

    bucket, key = s3a_path.replace("s3a://", "").split("/", 1)
    data = s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()
    return pq.ParquetFile(io.BytesIO(data)).metadata


def try_sql(spark, sql: str, label: str | None = None):
    """Run SQL and print OK / the first line of the error. For 'does this work here?' probes."""
    try:
        rows = spark.sql(sql).collect()
        print(f"OK     {label or sql[:70]}")
        return rows
    except Exception as exc:  # noqa: BLE001 - we want to show whatever the engine says
        first = str(exc).strip().splitlines()[0][:230]
        print(f"FAILED {label or sql[:70]}\n       -> {first}")
        return None


# --------------------------------------------------------------------------------------------------------------------
# dbt (dbt-core + dbt-trino are installed in the Jupyter image; the project talks to the same Trino as trino() above).
# Never name a Python module or package in the notebooks tree "dbt_*": dbt imports every importable module with that prefix as a plugin.
# --------------------------------------------------------------------------------------------------------------------
DBT_PROJECT = Path(__file__).resolve().parent / "dbt" / "lakehouse_demo"


def dbt(*args: str, project: Path = DBT_PROJECT, show_log: bool = False):
    """Run one dbt command in-process (dbtRunner) and return a DataFrame with one row per node.

    Example: dbt("build", "--select", "fct_orders"). Errors of failed nodes are printed in full.
    With show_log=True dbt's own console log is shown as well.
    """
    import pandas as pd
    from dbt.cli.main import dbtRunner

    argv = list(args) + ["--project-dir", str(project), "--profiles-dir", str(project), "--log-path", str(project / "logs"),
                         "--no-use-colors"] + ([] if show_log else ["--quiet"])
    res = dbtRunner().invoke(argv)
    if res.exception is not None:
        raise RuntimeError(f"dbt {' '.join(args)} could not start: {res.exception}")
    if isinstance(res.result, list):                       # e.g. `dbt ls` prints its own listing; return it for programmatic use
        return res.result
    results = getattr(res.result, "results", None)
    if not results:
        print(f"dbt {' '.join(args)}: {'ok' if res.success else 'FAILED'}")
        return res
    rows = [{"node": r.node.name, "type": r.node.resource_type.value,
             "status": str(r.status.value if hasattr(r.status, "value") else r.status),
             "seconds": round(r.execution_time, 2), "message": (r.message or "")[:300]} for r in results]
    df = pd.DataFrame(rows)
    bad = df[df.status.isin(["error", "fail", "runtime error"])]
    print(f"dbt {' '.join(args)}: {len(df)} nodes, {int((df.status.isin(['success', 'pass'])).sum())} ok, {len(bad)} not ok")
    for _, r in bad.iterrows():
        print(f"  [{r.status}] {r.node}: {r.message}")
    return df


DBT_SCHEMAS = ("dbt_demo", "dbt_demo_raw", "dbt_demo_snapshots", "dbt_test__audit")


def reset_dbt_demo():
    """Drop every table and view the dbt demo created, for a clean start. The schemas themselves are kept on purpose:
    in this lab a schema that was dropped cannot be created again under the same name (Hive Metastore fails with
    'Unable to create database managed directory' because an empty managed/<name>.db directory stays in SeaweedFS)."""
    existing = set(trino("SHOW SCHEMAS").iloc[:, 0])
    dropped = 0
    for schema in DBT_SCHEMAS:
        if schema not in existing:
            continue
        objs = trino(f"SELECT table_name, table_type FROM information_schema.tables WHERE table_schema = '{schema}'")
        for name, kind in objs.itertuples(index=False):
            trino(f'DROP {"VIEW" if kind == "VIEW" else "TABLE"} IF EXISTS iceberg.{schema}."{name}"')
            dropped += 1
    print(f"dbt demo reset: {dropped} tables/views dropped (schemas kept)")


# ---------------------------------------------------------------------------------------------------------------
# Chat with your data: the Kubeflow Semantic Operator (Helm chart only, semantic.enabled=true)
# ---------------------------------------------------------------------------------------------------------------
SEMANTIC_USER = "lab"
# Generation cap per chat turn (a model that keeps reasoning otherwise never stops: a real run grew an Ollama runner to
# 7 GB and got OOM-killed); a tool call or a SQL answer needs far less.
MAX_TOKENS = 1024


def semantic_available() -> bool:
    return bool(os.environ.get("SEMANTIC_MCP_URL"))


def require_semantic() -> None:
    if not semantic_available():
        raise RuntimeError("The semantic layer is off in this deployment: set semantic.enabled=true in the chart values.")
    if os.environ.get("SEMANTIC_AUTH_MODE", "header") != "header":
        raise RuntimeError("The semantic server runs with semantic.server.auth.mode=jwt; these helpers send identity "
                           "headers (X-Semantic-User / X-Semantic-Role) and need header mode.")


def _mcp_post(url: str, headers: dict, body: dict):
    """POST one JSON-RPC message (MCP streamable HTTP). Keeps the session id in `headers`; None for notifications."""
    import urllib.request

    req = urllib.request.Request(url, json.dumps(body).encode(), headers)
    with urllib.request.urlopen(req, timeout=120) as r:
        if r.headers.get("mcp-session-id"):
            headers["mcp-session-id"] = r.headers["mcp-session-id"]
        raw = r.read().decode()
    if "id" not in body or not raw.strip():
        return None
    if raw.lstrip().startswith("{"):
        return json.loads(raw)
    return json.loads([line[5:] for line in raw.splitlines() if line.startswith("data:")][-1])


def _mcp(method: str, params: dict, role: str):
    """One semantic-layer MCP session (identity headers for `role`). Returns the JSON-RPC result."""
    return _mcp_session(os.environ["SEMANTIC_MCP_URL"], method, params,
                        {"X-Semantic-User": SEMANTIC_USER, "X-Semantic-Role": role})


def _mcp_session(url: str, method: str, params: dict, extra_headers: dict | None = None):
    """One MCP session: initialize, then `method`. Returns the JSON-RPC result."""
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
               **(extra_headers or {})}
    _mcp_post(url, headers, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "data-lab-notebook", "version": "1"}}})
    _mcp_post(url, headers, {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
    res = _mcp_post(url, headers, {"jsonrpc": "2.0", "id": 2, "method": method, "params": params})
    if "error" in res:
        raise RuntimeError(f"MCP {method}: {res['error']}")
    return res["result"]


def mcp_call(name: str, args: dict, role: str = "analyst") -> dict:
    """Call one semantic-layer tool. Tool errors (e.g. a governance refusal) come back as {"error": text}."""
    res = _mcp("tools/call", {"name": name, "arguments": args}, role)
    text = "".join(c.get("text", "") for c in res.get("content", []) if c.get("type") == "text")
    if res.get("isError"):
        return {"error": text}
    return res.get("structuredContent") or json.loads(text or "{}")


_SUBSCHEMA_KEYS = ("items", "additionalItems", "contains", "not")   # where `true` means "any value" (a schema)


def _tool_schema(schema):
    """A copy of a JSON Schema that Ollama can compile into a tool-call grammar: every object gets `properties`
    (Ollama 0.40: "properties must be an object") and boolean sub-schemas (`"items": true`) become `{}`. Everything
    else (nullable unions such as ["null", "array"] included, which Ollama accepts) is kept as the server sent it."""
    if isinstance(schema, list):
        return [_tool_schema(s) for s in schema]
    if not isinstance(schema, dict):
        return schema
    out = {k: ({} if v is True and k in _SUBSCHEMA_KEYS else _tool_schema(v)) for k, v in schema.items() if k != "properties"}
    if out.get("type") == "object":
        out["properties"] = {k: _tool_schema(v) for k, v in (schema.get("properties") or {}).items()}
    return out


def mcp_tool_to_openai(tool: dict) -> dict:
    return {"type": "function", "function": {"name": tool["name"], "description": tool.get("description", ""),
                                             "parameters": _tool_schema(tool.get("inputSchema") or {"type": "object"})}}


def mcp_tools(role: str = "analyst") -> list:
    """The semantic layer's MCP tools as Ollama tool schemas."""
    return [mcp_tool_to_openai(t) for t in _mcp("tools/list", {}, role)["tools"]]


def wait_for_model(model_id: str, timeout_s: int = 180) -> None:
    """Wait until the semantic layer serves `model_id` (the operator publishes it once its tables exist)."""
    deadline = time.time() + timeout_s
    while True:
        names = [m["name"] for m in mcp_call("list_models", {}).get("models", [])]
        if model_id in names:
            return
        if time.time() >= deadline:
            raise TimeoutError(f"semantic model {model_id!r} not published after {timeout_s}s (served: {names})")
        time.sleep(5)


# ---------------------------------------------------------------------------------------------------------------
# LLM access: one OpenAI-compatible API (a hosted or self-hosted endpoint, or the chart's Ollama at /v1), or none.
# The chart sets LLM_PROVIDER (ollama | openai | none), LLM_BASE_URL, LLM_CHAT_MODEL, LLM_TOOL_MODEL, LLM_EMBED_MODEL
# and, from a Secret, LLM_API_KEY. Embeddings may live elsewhere: LLM_EMBED_BASE_URL / LLM_EMBED_API_KEY.
# ---------------------------------------------------------------------------------------------------------------
OLLAMA_TOOL_MODEL = "qwen3:4b-instruct"     # not plain `qwen3:4b`: that tag is now the thinking-only 2507 model


def llm_provider() -> str:
    return os.environ.get("LLM_PROVIDER", "").strip().lower() or "none"


def llm_available() -> bool:
    return llm_provider() in ("ollama", "openai")


def chat_model() -> str:
    return os.environ.get("LLM_CHAT_MODEL", "")


def tool_model() -> str:
    return os.environ.get("LLM_TOOL_MODEL", "") or (OLLAMA_TOOL_MODEL if llm_provider() == "ollama" else "")


def embed_model() -> str:
    return os.environ.get("LLM_EMBED_MODEL", "")


def embeddings_available() -> bool:
    return llm_available() and bool(embed_model())


def require_llm() -> None:
    if not llm_available():
        raise RuntimeError("No LLM is configured (LLM_PROVIDER=" + llm_provider() + "): set the chart's llm values "
                           "(llm.provider=openai with a base URL, models and key Secret, or ollama.enabled=true).")


def skip_unless_llm(section: str) -> bool:
    """For the AI-off path: prints one `skipped (no LLM configured): <section>` line and returns False when no LLM is set."""
    if llm_available():
        return True
    print(f"skipped (no LLM configured): {section}. Set the chart's llm values to run it.")
    return False


def openai_client():
    import openai

    return openai.OpenAI(base_url=endpoint("LLM_BASE_URL"), api_key=os.environ.get("LLM_API_KEY") or "unused",
                         max_retries=1, timeout=300)


def embeddings_config() -> dict:
    """Endpoint, key and model of the embeddings. Same endpoint and key as chat unless LLM_EMBED_BASE_URL is set; the
    chat key is never sent to a different endpoint."""
    base = os.environ.get("LLM_EMBED_BASE_URL") or ""
    if base and base != os.environ.get("LLM_BASE_URL"):
        key = os.environ.get("LLM_EMBED_API_KEY") or "unused"
    else:
        base = endpoint("LLM_BASE_URL")
        key = os.environ.get("LLM_EMBED_API_KEY") or os.environ.get("LLM_API_KEY") or "unused"
    return {"base_url": base, "api_key": key, "model": embed_model()}


def embeddings_client():
    import openai

    c = embeddings_config()
    return openai.OpenAI(base_url=c["base_url"], api_key=c["api_key"], max_retries=1, timeout=300)


def embed(texts: list) -> list:
    """Vectors for `texts`, in order."""
    out = embeddings_client().embeddings.create(model=embed_model(), input=list(texts))
    return [d.embedding for d in out.data]


def skip_unless_embeddings(section: str) -> bool:
    """Like skip_unless_llm, for sections that also need an embeddings model (llm.embeddingModel)."""
    if not skip_unless_llm(section):
        return False
    if not embed_model():
        print(f"skipped (no embeddings model configured): {section}. Set llm.embeddingModel "
              "(and llm.embeddingBaseUrl when another endpoint serves it).")
        return False
    return True


def langchain_chat(temperature: float = 0.1, model: str | None = None):
    """A LangChain chat model on the lab's LLM endpoint (traced by the OpenInference LangChain instrumentor)."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(base_url=endpoint("LLM_BASE_URL"), api_key=os.environ.get("LLM_API_KEY") or "unused",
                      model=model or chat_model(), temperature=temperature, max_tokens=MAX_TOKENS)


def langchain_embeddings():
    """LangChain embeddings on the embeddings endpoint (chat's unless LLM_EMBED_BASE_URL is set)."""
    from langchain_openai import OpenAIEmbeddings

    c = embeddings_config()
    # check_embedding_ctx_length=False sends plain strings: non-OpenAI servers do not accept token arrays
    return OpenAIEmbeddings(base_url=c["base_url"], api_key=c["api_key"], model=c["model"], check_embedding_ctx_length=False)


def _llm_error(e: Exception, model: str) -> RuntimeError:
    import openai

    where = f"LLM_BASE_URL={os.environ.get('LLM_BASE_URL', '?')}, model {model}"
    if isinstance(e, openai.APIStatusError):
        return RuntimeError(f"LLM request failed with HTTP {e.status_code} ({where}): {str(e.message)[:300]}")
    return RuntimeError(f"LLM request failed ({where}): {type(e).__name__}: {str(e)[:300]}")


def _chat(model: str, messages: list, tools=None, max_tokens: int = MAX_TOKENS) -> dict:
    """One chat turn. Returns {"message": {"content", "tool_calls": [{"function": {"name", "arguments": dict}}]}}."""
    import openai

    kwargs = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0}
    if tools:
        kwargs["tools"] = tools
    try:
        reply = openai_client().chat.completions.create(**kwargs).choices[0].message
    except openai.OpenAIError as e:
        raise _llm_error(e, model) from None
    calls = [{"id": tc.id, "function": {"name": tc.function.name, "arguments": json.loads(tc.function.arguments or "{}")}}
             for tc in (reply.tool_calls or [])]
    return {"message": {"content": reply.content or "", "tool_calls": calls}}


def require_tool_model(model: str | None = None) -> None:
    """Stop early, with the model's name, when the endpoint rejects tool calling for it (one tiny probe request)."""
    import openai

    model = model or tool_model()
    probe = [{"type": "function", "function": {"name": "ping", "description": "probe",
                                               "parameters": {"type": "object", "properties": {}}}}]
    try:
        openai_client().chat.completions.create(model=model, messages=[{"role": "user", "content": "ping"}],
                                                tools=probe, max_tokens=16)
    except openai.OpenAIError as e:
        raise RuntimeError(f"model {model} does not support tool calling at this endpoint (set llm.toolModel to one "
                           f"that does): {_llm_error(e, model)}") from None




# ---------------------------------------------------------------------------------------------------------------
# Operations agent: the Airflow and Spark History MCP servers (Helm chart only: airflowMcp / sparkHistoryMcp)
# ---------------------------------------------------------------------------------------------------------------
OPS_MCP = {"airflow": "AIRFLOW_MCP_URL", "spark_history": "SPARK_HISTORY_MCP_URL"}   # server -> env var (endpoints)
# The read-only tools the agent is offered. All tools of both servers (51) are ~15k tokens of schemas, beyond Ollama's
# default 4,096-token context; nothing here can trigger, pause, clear or delete.
OPS_TOOLS = {
    "airflow": {"list_dags", "get_dag_details", "list_dag_runs", "get_dag_run", "diagnose_dag_run", "list_tasks",
                "get_task_instance", "get_task_logs", "list_import_errors", "get_system_health"},
    "spark_history": {"list_applications", "list_jobs", "list_stages", "get_stage", "get_job_bottlenecks",
                      "list_sql_executions", "get_executor_summary", "get_sql_execution", "compare_sql_executions",
                      "compare_job_performance", "compare_job_environments"},
}


def ops_servers() -> dict:
    """The operations MCP servers this deployment runs: {name: url}."""
    return {name: os.environ[env] for name, env in OPS_MCP.items() if os.environ.get(env)}


def require_ops() -> None:
    if not ops_servers():
        raise RuntimeError("The Airflow and Spark History MCP servers run only on the Helm chart (airflowMcp.enabled, "
                           "sparkHistoryMcp.enabled; the workstation profile turns both on).")


def ops_tools(server: str) -> list:
    return _mcp_session(ops_servers()[server], "tools/list", {})["tools"]


def ops_call(server: str, name: str, args: dict):
    """Call one tool on an operations MCP server. Returns parsed JSON when the tool returns JSON, else the text."""
    res = _mcp_session(ops_servers()[server], "tools/call", {"name": name, "arguments": args})
    text = "".join(c.get("text", "") for c in res.get("content", []) if c.get("type") == "text")
    if res.get("isError"):
        return {"error": text}
    if res.get("structuredContent") is not None:
        return res["structuredContent"]
    try:
        return json.loads(text)
    except ValueError:
        return text


RAW_PROMPT = ("You write one Trino SQL query that answers the question over the tables below (catalog {catalog}, "
              "schema {schema}). Reply with the query only, in a ```sql block.\n\n{ddl}")


def _table_ddl(tables: list, schema: str) -> str:
    """CREATE-style column lists from information_schema (works for dbt's views too; SHOW CREATE TABLE does not)."""
    names = ", ".join(f"'{t}'" for t in tables)
    cols = trino(f"SELECT table_name, column_name, data_type FROM {TRINO_CATALOG}.information_schema.columns "
                 f"WHERE table_schema = '{schema}' AND table_name IN ({names}) ORDER BY table_name, ordinal_position")
    by_table = {}
    for table, column, dtype in cols.values.tolist():
        by_table.setdefault(table, []).append(f"  {column} {dtype}")
    return "\n\n".join(f"CREATE TABLE {TRINO_CATALOG}.{schema}.{t} (\n" + ",\n".join(by_table.get(t, [])) + "\n)"
                       for t in tables)


def ask_raw_sql(question: str, tables: list, model: str | None = None, schema: str = "dbt_demo") -> dict:
    """The baseline: the LLM sees the tables' columns and writes Trino SQL itself."""
    prompt = RAW_PROMPT.format(catalog=TRINO_CATALOG, schema=schema, ddl=_table_ddl(tables, schema))
    reply = _chat(model or tool_model(), [{"role": "system", "content": prompt}, {"role": "user", "content": question}])["message"]
    text = re.sub(r"<think>.*?</think>", "", reply.get("content", ""), flags=re.S)   # thinking models may reason inline
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    sql = (m.group(1) if m else text).strip().rstrip(";")
    try:
        df = trino(sql)
        return {"answer": text, "sql": sql, "rows": df.values.tolist(), "error": None}
    except Exception as e:  # the model's SQL failed: that is part of the comparison
        return {"answer": text, "sql": sql, "rows": [], "error": str(e)}


def _norm(v, tol):
    if hasattr(v, "isoformat"):                       # date / datetime / pandas Timestamp: compare by day
        return v.isoformat()[:10]
    text = str(v).strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}([T ]|$)", text):   # ISO date or timestamp string (e.g. from the semantic server)
        return text[:10]
    try:
        return round(float(v) / tol) * tol
    except (TypeError, ValueError):
        return text.lower()


def rows_match(rows, truth, tol: float = 0.01) -> bool:
    """Same result set as the ground truth: row and column order, number types and rounding to `tol` do not matter."""
    def canon(rs):
        return sorted(tuple(sorted((_norm(v, tol) for v in r), key=repr)) for r in rs)
    return bool(truth) and len(rows) == len(truth) and canon(rows) == canon(truth)
