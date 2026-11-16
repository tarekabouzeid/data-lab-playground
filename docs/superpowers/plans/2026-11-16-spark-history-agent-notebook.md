# Spark History agent notebook (`ops/02`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A chart-only notebook, `jupyter/notebooks/ops/02_spark_history_agent.ipynb`, that shows what the Spark History MCP server can do. It runs a small Spark workload, inspects it with the MCP tools by hand, then lets the tool model diagnose it.

**Architecture:**
- The lab already deploys the server: `sparkHistoryMcp.enabled` in `charts/data-lab/ci/workstation-values.yaml`, service `…-spark-history-mcp:18888/mcp`. The notebook reaches it through `lab_utils` (`SPARK_HISTORY_MCP_URL`).
- One MCP endpoint serves two Spark backends from `config.yaml`:
  - `history`: the History Server, for completed apps.
  - `live`: the Spark Connect UI on :4040, for the running session.
- The notebook uses `live` for the queries it just ran. On S3A, a running app's events only reach `s3a://warehouse/spark-events` when it stops (rolling `eventlog_v2`, found 2026-11-16), so the History Server cannot list the current session.
- The only code change outside the notebook is an optional `servers=` filter on `lab_utils.ask_ops`, plus a few more read-only Spark tools.

**Tech Stack:**
- Spark 4.1.3 Connect
- Iceberg 1.12 tables from `lab_data` (`shop_bronze`, `dbt_demo`)
- Spark History MCP 1.0.1 (`mcp-apache-spark-history-server`)
- Ollama `qwen3:4b-instruct`
- pytest

**Spec:** the proposal in the 2026-11-16 session, summarized here:
1. A workload: the same scan with and without a partition filter, and one deliberately skewed join.
2. The tools by hand on `live`: `list_jobs`/`list_stages`, `get_sql_execution`, `get_job_bottlenecks`, `compare_sql_executions`.
3. The tools on `history`: `list_applications`, `compare_job_environments`.
4. The agent, given only the Spark tools: "Why was my last query slow?", "Which stage had the most skew?", "How much less data did the filtered query read?", with every tool call printed.
5. "What to look for": the server is read-only, the S3 in-progress caveat, and small models sometimes pick the wrong server.

## Global Constraints

- Chart only. Docker Compose is frozen; the notebook stops early on Compose with `lu.require_ops()`-style guidance.
- Notebook rules (CLAUDE.md "Notebooks Location"):
  - no hardcoded hosts (`tests/unit/test_notebook_endpoints.py` enforces this);
  - no `.master()`, no `SparkContext`/RDDs (Spark Connect);
  - never `DROP SCHEMA`;
  - no module named `dbt_*`.
- No version changes. No chart change unless Task 0 proves one is needed; if it is, bump the chart version, add a CHANGELOG entry, and regenerate the schema and helm-docs (Post-Change Rule 4).
- Memory: run in the workstation lab with the memory guard on. The lab must stay under about 18.5 GiB, with other kind clusters stopped.
- LLM: `lab_utils.TOOL_MODEL` (`qwen3:4b-instruct`). Every `_chat` caps `num_predict` (AGENTS.md pitfall 18a). Keep the offered tool schemas within `OPS_NUM_CTX = 16384`.
- Post-Change Rule: update README.md, AGENTS.md, CLAUDE.md and `docs/WORKSTATION_TESTING.md`. Commit each logical change separately and push to `claude/upgrade-iceberg-spark-hive-trino-yc0p1s`. No PR, no `RELEASE_NOTES.md`.

## Review Focus

1. **`live` has a new app id after Spark Connect restarts** (pod restart or `helm upgrade`). The notebook must list applications on `live` first and never hardcode an id. Task 3 asserts this.
2. **The History Server has no completed apps** on a fresh lab, or after the 7-day cleaner (`spark.history.fs.cleaner.maxAge 7d`). The history section must print an explanation and carry on, not raise. Task 3 asserts this.
3. **The model calls a tool on the wrong server**, e.g. `history` for the live session, and gets "not found". The prompt must say which server holds what. Task 1 pins the prompt.
4. **Huge tool results** (stage lists, SQL plans) overflow the context. Results stay capped at 8,000 characters, as `ask_ops` already does. Task 1 has a test.
5. **Only one ops server is enabled** (`airflowMcp` on, `sparkHistoryMcp` off). The notebook must stop with a clear message. Task 1 has a test.

---

### Task 0: Probe the live MCP server (no code, findings into this plan)

**Files:** none. Append the findings to this plan under "Task 0 findings".

- [ ] **Step 1:** In the lab's Jupyter pod, run `lu.ops_tools("spark_history")` and write down each tool's `inputSchema`. Expected: most tools take an optional `server` argument (`"history"` default, `"live"`) and an `app_id`. Confirm the exact argument names.
- [ ] **Step 2:** Run `lu.ops_call("spark_history", "list_applications", {"server": "live"})`. Expected: one app, `Spark Connect server`.
- [ ] **Step 3:** Run a query over Spark Connect, then call `list_sql_executions` and `get_sql_execution` on `live`. Check whether the plan text is present. The chart sets `include_plan_description: false`. If the plan text is missing, the notebook shows the plan with `df.explain()` instead; do not change the chart for it.
- [ ] **Step 4:** Call `get_job_bottlenecks` and `compare_sql_executions` on `live`, and record their argument names and output shape.
- [ ] **Step 5:** Pick the workload tables. Read `lab_data.py` for what exists after `iceberg/00–04` and `dbt/01–03` (e.g. `shop_bronze.order_events`, partitioned how?). Choose:
  - the partitioned table and filter column for the pruning comparison;
  - a skewed join key. A good candidate is `order_events` joined to itself or to a 1-row dimension on a hot key; if the data has no natural skew, build the skewed side with `spark.range` plus a constant key.

  Write the choice down.

### Task 1: `ask_ops` can be limited to some servers; more read-only Spark tools

**Files:**
- Modify: `jupyter/notebooks/lab_utils.py` (`OPS_TOOLS`, `OPS_PROMPT`, `ask_ops`)
- Test: `tests/unit/test_lab_utils_semantic.py`

**Interfaces:**
- Produces: `ask_ops(question: str, model: str = TOOL_MODEL, max_turns: int = 8, servers: tuple[str, ...] | None = None) -> dict`.
  - With `servers=None` (the default), it behaves as today.
  - Otherwise it offers only tools of those servers. A server that isn't running raises `RuntimeError` naming the missing server and its chart value.
- Produces: `OPS_TOOLS["spark_history"]` also contains `"get_sql_execution"`, `"compare_sql_executions"`, `"compare_job_performance"` and `"compare_job_environments"`. All are read-only; correct the names if Task 0 finds different ones.
- Produces: `OPS_PROMPT` adds one sentence: the Spark tools take `server="live"` for the running Spark Connect session and `server="history"` for finished applications.

- [ ] **Step 1: Write the failing tests**

```python
def test_ops_agent_can_be_limited_to_spark_history(monkeypatch):
    # both servers configured; servers=("spark_history",) offers only spark_history__* tools
    ...
    lab_utils.ask_ops("why slow?", servers=("spark_history",))
    assert offered and all(n.startswith("spark_history__") for n in offered)

def test_ops_agent_names_a_missing_server(monkeypatch):
    monkeypatch.setenv("AIRFLOW_MCP_URL", "http://a/mcp"); monkeypatch.delenv("SPARK_HISTORY_MCP_URL", raising=False)
    with pytest.raises(RuntimeError, match="sparkHistoryMcp.enabled"):
        lab_utils.ask_ops("q", servers=("spark_history",))

def test_ops_prompt_says_which_spark_server_holds_what():
    assert 'server="live"' in lab_utils.OPS_PROMPT and 'server="history"' in lab_utils.OPS_PROMPT

def test_ops_agent_caps_tool_results(monkeypatch):
    # a 50,000-character tool result reaches the model as at most 8,000 characters
```

  Mock with the same `_mcp_session`/`_chat` fakes as `test_ops_agent_offers_a_curated_read_only_tool_set`. Also update that test's expected Spark set if it asserts on Spark tool names.
- [ ] **Step 2:** Run `python -m pytest tests/unit/test_lab_utils_semantic.py -q`. Expected: the 3 new behaviour tests FAIL. The cap test may already pass; keep it as a guard.
- [ ] **Step 3:** Implement the `servers` filter, the extra tool names and the prompt sentence.
- [ ] **Step 4:** Run `python -m pytest tests/chart tests/unit -q`. Expected: all pass. The previous count was 126.
- [ ] **Step 5:** Commit: `notebooks: ask_ops can be limited to one ops server; Spark compare tools`.

### Task 2: The notebook

**Files:**
- Create: `jupyter/notebooks/ops/02_spark_history_agent.ipynb`

**Interfaces:**
- Consumes: `lu.require_ops()`, `lu.require_tool_model()`, `lu.ops_servers()`, `lu.ops_call(server, name, args)`, and `lu.ask_ops(q, servers=("spark_history",))` from Task 1.
- Consumes: the Spark session as in the other chart notebooks (`SparkSession.builder.remote(os.environ["SPARK_REMOTE"])` via the existing `lab_utils` helper; copy the pattern from `iceberg/00`).

Cells, in this order. Every markdown cell gets a short "why":

1. **Title and intro** ("Stage 9b, observe Spark"). Covers the two backends (`live`, `history`), the S3 in-progress caveat in one sentence, and the note that the server is read-only.
2. **Setup.** Run `lu.require_ops()`; stop with a clear message if `"spark_history" not in lu.ops_servers()`. Then `lu.require_tool_model()`, get the Spark session, and print the tool list.
3. **Workload.** Tag each query with `spark.sparkContext`-free job descriptions. Use `spark.conf.set("spark.job.description", …)` if Connect allows it; otherwise rely on SQL ids from `list_sql_executions` (Task 0). Run:
   - (a) the scan with the partition filter, `.count()`;
   - (b) the same scan without the filter;
   - (c) the skewed join, `.count()`.

   Record wall times with `time.perf_counter()`.
4. **Tools by hand, `live`:**
   - Find the live app id with `list_applications(server="live")`.
   - Show `list_sql_executions` and pick the three executions by description or order.
   - Show `get_sql_execution` for (a): plan text, or `df.explain()` if Task 0 found it missing.
   - Show `compare_sql_executions` for (a) against (b): input bytes and rows read.
   - Show `get_job_bottlenecks` for (c): the skewed stage, with max and median task time.

   Render each result as a small pandas table, not raw JSON.
5. **Tools by hand, `history`:** run `list_applications(server="history")`.
   - If it is empty, print why: the cleaner at 7 days, and the live session only appears after Spark Connect stops.
   - If there are two or more apps, run `compare_job_environments` on two of them.
6. **Agent.** Use `investigate(q)`, the same helper shape as `ops/01`, with `servers=("spark_history",)`, on three questions:
   - "Why was my last Spark query slow? Name the stage."
   - "Which stage had the most task skew in the running Spark Connect session?"
   - "How much less data did the filtered scan read than the unfiltered one?"
7. **What to look for:**
   - the expected answers (the skewed stage id from step 4; a bytes-read ratio of roughly X from Task 0);
   - the server is read-only;
   - the S3 caveat;
   - if the model asks `history` for the live app, rerun or name the server in the question.

- [ ] **Step 1:** Write the notebook with no outputs. Strip outputs before committing.
- [ ] **Step 2:** Run `python -m pytest tests/unit/test_notebook_endpoints.py -q`. Expected: PASS (no hardcoded hosts).
- [ ] **Step 3:** Run it headless in the lab with the memory guard on:
  - `kubectl -n datalab exec deploy/datalab-data-lab-jupyter -- jupyter nbconvert --to notebook --execute work/ops/02_spark_history_agent.ipynb --output /tmp/v_02.ipynb`

  Expected: no errors. Step 4 names the skewed stage, and the agent's answers match step 4's numbers.
- [ ] **Step 4:** Run it again after `kubectl -n datalab rollout restart deploy/datalab-data-lab-spark-connect`. This is Review Focus 1: the new app id is picked up, and the history section now lists the old session.
- [ ] **Step 5:** Commit: `notebooks: ops/02 Spark History agent (live + history MCP)`.

### Task 3: e2e and docs

**Files:**
- Modify: `hack/workstation/e2e.sh` (near line 143). Add `run_notebook ops/02_spark_history_agent.ipynb` when `endpoint SPARK_HISTORY_MCP_URL` is set, otherwise `warn … skipping`.
- Modify: `docs/WORKSTATION_TESTING.md` (the notebook table, near line 98). Add a row: "`ops/02_spark_history_agent.ipynb`: names the skewed stage; the filtered scan reads less; the steps table shows only `spark_history__*` tools."
- Modify: README.md (the notebooks list), AGENTS.md (the notebooks entry near line 294, plus one line in pitfall 13/19 on the S3 in-progress event logs), and CLAUDE.md (the "Notebooks Location" ops line).
- Modify: `docs/TODO_MANUAL_TESTS.md`. Tick the "Next" item, and add the browser check to "Already verified".

- [ ] **Step 1:** Edit `e2e.sh`, then run `shellcheck hack/workstation/e2e.sh`. Expected: clean.
- [ ] **Step 2:** Make the doc edits. Then run `python -m pytest tests/chart tests/unit -q` and `hack/check-compose-parity.py`. Expected: all pass, parity 0.
- [ ] **Step 3:** In the browser (Playwright MCP, local scope), open the notebook in JupyterLab under `work/ops/`, choose Run All, and check the three answers. Afterwards restore the notebook with `git checkout`: autosave writes outputs into the repo mount.
- [ ] **Step 4:** Commit `e2e: run ops/02 when the Spark History MCP is on` and `docs: ops/02 Spark History agent`, then push.

## Task 0 findings

(to fill in)
