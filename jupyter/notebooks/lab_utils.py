"""Helpers shared by the technique notebooks (iceberg/, dbt/).

Works unchanged on both deployments of the lab:
  * Docker Compose: classic Spark. The session is created here with the Iceberg REST catalog; the master and the S3A
    settings come from $SPARK_HOME/conf/spark-defaults.conf in the Jupyter image.
  * Kubernetes (charts/data-lab): Spark Connect (SPARK_REMOTE is set). The Spark Connect server already has the catalog,
    the Iceberg SQL extensions and the S3A settings (charts/data-lab/templates/spark-configmap.yaml). Do not call .master() here:
    pyspark 4.1 rejects it together with SPARK_REMOTE, and static confs cannot be set by a Connect client.

Every service endpoint lives here: an environment variable set by the Helm chart, with the Docker Compose host as the
default. Notebooks import these constants instead of hardcoding hosts (tests/unit/test_notebook_endpoints.py).

Imports of optional packages (boto3, trino, pyarrow) are lazy so the module imports anywhere.
"""
import json
import os
import re
import time
from pathlib import Path

CATALOG = "iceberg_catalog"                 # Spark name of the HMS Iceberg REST catalog
TRINO_CATALOG = "iceberg"                   # the same catalog as seen by Trino
REST_URI = os.environ.get("ICEBERG_REST_URI", "http://hive-metastore:9084/iceberg")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "http://seaweedfs:8333")
S3_KEY = os.environ.get("S3_ACCESS_KEY", "seaweedadmin")
S3_SECRET = os.environ.get("S3_SECRET_KEY", "seaweedadmin123")
TRINO_HOST = os.environ.get("TRINO_HOST", "trino")
TRINO_PORT = int(os.environ.get("TRINO_PORT", "8080"))
OLLAMA_URL = os.environ.get("OLLAMA_URL", os.environ.get("OLLAMA_HOST", "http://ollama:11434"))
QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY") or None   # the chart enables Qdrant API keys; Compose does not
PHOENIX_URL = os.environ.get("PHOENIX_URL", "http://phoenix:6006")
PHOENIX_OTLP_ENDPOINT = os.environ.get("PHOENIX_OTLP_GRPC_ENDPOINT", "http://phoenix:4317")


def on_spark_connect() -> bool:
    return bool(os.environ.get("SPARK_REMOTE"))


def get_spark(app_name: str = "lab-notebook"):
    """A Spark session wired to the Iceberg catalog, on either deployment."""
    from pyspark.sql import SparkSession

    builder = SparkSession.builder.appName(app_name)
    if not on_spark_connect():
        builder = (
            builder.config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
            .config(f"spark.sql.catalog.{CATALOG}", "org.apache.iceberg.spark.SparkCatalog")
            .config(f"spark.sql.catalog.{CATALOG}.type", "rest")
            .config(f"spark.sql.catalog.{CATALOG}.uri", REST_URI)
            .config(f"spark.sql.catalog.{CATALOG}.io-impl", "org.apache.iceberg.hadoop.HadoopFileIO")
        )
    return builder.getOrCreate()


def trino(sql: str):
    """Run one statement on Trino (catalog `iceberg`) and return a pandas DataFrame."""
    import pandas as pd
    import trino as trino_client

    conn = trino_client.dbapi.connect(host=TRINO_HOST, port=TRINO_PORT, user="lab", catalog=TRINO_CATALOG)
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    cols = [c.name for c in (cur.description or [])]
    return pd.DataFrame(rows, columns=cols)


def s3_client():
    import boto3

    return boto3.client("s3", endpoint_url=S3_ENDPOINT, aws_access_key_id=S3_KEY,
                        aws_secret_access_key=S3_SECRET, region_name="us-east-1")


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
TOOL_MODEL = "qwen3:4b"   # tool calling in Ollama (gemma3 has none)


def semantic_available() -> bool:
    return bool(os.environ.get("SEMANTIC_MCP_URL"))


def require_semantic() -> None:
    if not semantic_available():
        raise RuntimeError("The semantic layer runs only on the Helm chart with semantic.enabled=true "
                           "(Docker Compose is frozen and does not include it).")
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


def mcp_tool_to_ollama(tool: dict) -> dict:
    return {"type": "function", "function": {"name": tool["name"], "description": tool.get("description", ""),
                                             "parameters": tool.get("inputSchema", {"type": "object"})}}


def mcp_tools(role: str = "analyst") -> list:
    """The semantic layer's MCP tools as Ollama tool schemas."""
    return [mcp_tool_to_ollama(t) for t in _mcp("tools/list", {}, role)["tools"]]


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


def _ollama_client():
    import ollama

    return ollama.Client(host=OLLAMA_URL)


def _as_dict(obj):
    return obj.model_dump(exclude_none=True) if hasattr(obj, "model_dump") else obj


def _chat(model: str, messages: list, tools=None) -> dict:
    return _as_dict(_ollama_client().chat(model=model, messages=messages, tools=tools, options={"temperature": 0}))


def _ollama_show(model: str) -> dict:
    return _as_dict(_ollama_client().show(model))


def require_tool_model(model: str = TOOL_MODEL) -> None:
    if "tools" not in (_ollama_show(model).get("capabilities") or []):
        raise RuntimeError(f"Ollama model {model} has no tool calling; pull one that has (e.g. ollama pull {TOOL_MODEL})")


SEMANTIC_PROMPT = ("You answer business questions with the semantic layer's tools only. Find certified metrics with "
                   "list_metrics and group-by dimensions with list_dimensions, then call query_metric. Never invent "
                   "numbers and never write SQL. Reply in one or two sentences with the numbers from the result.")


def ask_semantic(question: str, model: str = TOOL_MODEL, role: str = "analyst", max_turns: int = 6) -> dict:
    """Answer through the semantic layer: the LLM picks certified metrics/dimensions, the server compiles the SQL.

    Returns the answer, the tool names called, the compiled SQL and rows of the last query_metric, and `steps`:
    every tool call with its arguments and the server's result, in order (the notebook walks through them)."""
    messages = [{"role": "system", "content": SEMANTIC_PROMPT}, {"role": "user", "content": question}]
    tools, calls, steps, sql, rows = mcp_tools(role), [], [], None, []
    for _ in range(max_turns):
        msg = _chat(model, messages, tools)["message"]
        messages.append(msg)
        if not msg.get("tool_calls"):
            return {"answer": msg.get("content", ""), "tool_calls": calls, "steps": steps, "sql": sql, "rows": rows}
        for tc in msg["tool_calls"]:
            name, args = tc["function"]["name"], tc["function"].get("arguments") or {}
            calls.append(name)
            out = mcp_call(name, args, role)
            steps.append({"tool": name, "args": args, "result": out})
            if name == "query_metric" and "sql" in out:
                sql, rows = out["sql"], out.get("rows", [])
            messages.append({"role": "tool", "tool_name": name, "content": json.dumps(out)[:8000]})
    return {"answer": f"(stopped after {max_turns} turns without a final answer)", "tool_calls": calls,
            "steps": steps, "sql": sql, "rows": rows}


# ---------------------------------------------------------------------------------------------------------------
# Operations agent: the Airflow and Spark History MCP servers (Helm chart only: airflowMcp / sparkHistoryMcp)
# ---------------------------------------------------------------------------------------------------------------
OPS_MCP = {"airflow": "AIRFLOW_MCP_URL", "spark_history": "SPARK_HISTORY_MCP_URL"}   # server -> env var (endpoints)


def ops_servers() -> dict:
    """The operations MCP servers this deployment runs: {name: url}."""
    return {name: os.environ[env] for name, env in OPS_MCP.items() if os.environ.get(env)}


def require_ops() -> None:
    if not ops_servers():
        raise RuntimeError("The Airflow and Spark History MCP servers run only on the Helm chart (airflowMcp.enabled, "
                           "sparkHistoryMcp.enabled; the workstation profile turns both on). Docker Compose is "
                           "frozen and does not include them.")


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


OPS_PROMPT = ("You are the on-call data engineer for Northwind Gear's lakehouse. Answer with the tools: the Airflow "
              "tools for DAGs, runs and task logs; the Spark History tools for Spark applications, jobs and stages. "
              "Never guess ids: list first, then drill down. Reply in at most five sentences with the facts you found.")


def ask_ops(question: str, model: str = TOOL_MODEL, max_turns: int = 8) -> dict:
    """Let the LLM investigate with the tools of every operations MCP server. Tool names are prefixed with the server
    (`airflow__list_dags`). Returns the answer and `steps` (every tool call, its arguments and a result excerpt)."""
    servers = ops_servers()
    tools, owner = [], {}
    for server in servers:
        for t in ops_tools(server):
            name = f"{server}__{t['name']}"
            owner[name] = (server, t["name"])
            tools.append(mcp_tool_to_ollama({**t, "name": name}))
    messages = [{"role": "system", "content": OPS_PROMPT}, {"role": "user", "content": question}]
    steps = []
    for _ in range(max_turns):
        msg = _chat(model, messages, tools)["message"]
        messages.append(msg)
        if not msg.get("tool_calls"):
            return {"answer": msg.get("content", ""), "steps": steps}
        for tc in msg["tool_calls"]:
            name, args = tc["function"]["name"], tc["function"].get("arguments") or {}
            if name in owner:
                out = ops_call(*owner[name], args)
            else:
                out = {"error": f"unknown tool {name}"}
            text = out if isinstance(out, str) else json.dumps(out, default=str)
            steps.append({"tool": name, "args": args, "result": text[:300]})
            messages.append({"role": "tool", "tool_name": name, "content": text[:8000]})
    return {"answer": f"(stopped after {max_turns} turns without a final answer)", "steps": steps}


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


def ask_raw_sql(question: str, tables: list, model: str = TOOL_MODEL, schema: str = "dbt_demo") -> dict:
    """The baseline: the LLM sees the tables' columns and writes Trino SQL itself."""
    prompt = RAW_PROMPT.format(catalog=TRINO_CATALOG, schema=schema, ddl=_table_ddl(tables, schema))
    reply = _chat(model, [{"role": "system", "content": prompt}, {"role": "user", "content": question}])["message"]
    text = re.sub(r"<think>.*?</think>", "", reply.get("content", ""), flags=re.S)   # qwen3 may think inline
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
