# Slimmer lab (no Compose, any LLM, semantic traces, Strands agents) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:**
- Remove Docker Compose.
- Let the lab run its AI parts against any OpenAI-compatible endpoint (or with AI off), so no GPU is needed.
- Send the semantic layer's traces to Phoenix.
- Build the ops agents on AWS Strands.

**Architecture:**
- The Helm chart is the only way to run the lab.
- One new `llm` values block becomes `LLM_*` env in Jupyter. Keys come only from an existing Secret.
- `lab_utils` talks to every model through the `openai` SDK (local Ollama serves the same API at `/v1`).
- The semantic server's own OTLP exporter is pointed at Phoenix.
- The notebooks trace their LLM and tool calls with OpenInference.
- `ops/01` and the new `ops/02` run Strands agents over the MCP servers.

**Tech Stack:**
- Helm 3.19 (`--kube-version 1.34.0`), pytest
- `openai` SDK, `langchain-openai`
- `strands-agents[openai]==1.65.0`
- `openinference-instrumentation-openai`
- Ollama 0.40.3
- Phoenix (OTLP over HTTP on :6006)

**Spec:** `docs/superpowers/specs/2026-11-17-slim-lab-llm-providers-design.md` (decisions D1–D8).

## Global Constraints

- `values.yaml` is a public API: add keys only, each with a `# --` doc comment. After any values change, run
  `python3 hack/gen-chart-schema.py` and `helm-docs -c charts -s file`.
- The default render is unchanged: `tests/chart/golden/default-resources.txt` must still pass. Every new feature is off,
  or inert, by default.
- Chart version: `0.3.1` → `0.4.0`, with a `charts/data-lab/CHANGELOG.md` entry. `appVersion` stays `3.0.0` (the release
  decision is the maintainer's, D8).
- No key in values, ConfigMaps, notebooks or logs. LLM keys come from `llm.auth.existingSecret` / `llm.embeddings.auth.existingSecret` (key name `api-key` by default).
- `LLM_PROVIDER` values are exactly `ollama`, `openai`, `none`. Chart default `llm.provider: ""` means auto (`ollama` when `ollama.enabled`, else `none`).
- Ollama defaults (D2): chat `gemma3:4b`, tool `qwen3:4b-instruct`, embeddings `mxbai-embed-large:latest`, `ollama.contextLength: 16384`.
- Notebook rules:
  - no hardcoded hosts (`tests/unit/test_notebook_endpoints.py`);
  - no `.master()`, `SparkContext` or RDDs;
  - no `DROP SCHEMA`;
  - no module named `dbt_*`.
- AI-off notebooks must run to the end headless. A skipped section prints a line starting `skipped (no LLM configured):`.
- Keep `lab_utils` guards: schema normalizer, model-name guard, grain guard, curated ops tool set, today's date in the
  ops prompt, `max_tokens` capped on every chat call (`MAX_TOKENS = 1024`).
- Phoenix projects: notebook LLM spans `semantic-layer` (semantic/01), `ops-agents` (ops/01, ops/02), `northwind-rag`
  (rag_demo). Server spans land in `default` (upstream sets only `service.name`).
- Post-Change Rule: README, AGENTS, CLAUDE, VERSIONS, WORKSTATION_TESTING, TODO_MANUAL_TESTS updated in the task that
  changes the behavior. Commit per task and push to `claude/upgrade-iceberg-spark-hive-trino-yc0p1s`. No
  `RELEASE_NOTES.md`.
- Verify each task with `python3 -m pytest tests/chart tests/unit -q`. Lint each profile with
  `helm lint --strict charts/data-lab --kube-version 1.34.0 -f <profile> --set global.apiChecks=false`.

## Review Focus

1. **`openai` mode with a missing or wrong Secret**: expect a clear render-time `fail` when `existingSecret` is empty,
   and at runtime a one-line error naming `LLM_BASE_URL` and the HTTP status, not a stack trace. Tests: Task 2
   `test_openai_requires_secret`; Task 4 `test_chat_error_names_endpoint`.
2. **A tool model without tool calling** (e.g. a chat-only OpenRouter model): expect `require_tool_model()` to say so
   once, and the agent cells to skip instead of looping. Test: Task 4 `test_require_tool_model_probe_rejects_no_tools`.
3. **Embeddings unset while chat is set** (OpenRouter): expect `rag_demo` section 1 to run and sections 2–3 to print the
   reason. Tests: Task 4 `test_embeddings_available_false_without_model`; Task 6 notebook check.
4. **Phoenix disabled, or auth on without a headers Secret**: expect no `otlpEndpoint` in the server config and a
   NOTES.txt line. Notebook tracing must never fail a cell. Tests: Task 3 `test_no_tracing_when_phoenix_off`,
   `test_auth_without_headers_secret_disables_tracing`; Task 4 `test_tracing_setup_never_raises`.
5. **Notebook opened outside the chart** (no endpoint env): expect one clear message to run it in the lab's Jupyter, not
   a connection to a Compose host name. Test: Task 1 `test_missing_endpoint_env_raises_clear_error`.

---

### Task 1: Remove Docker Compose

**Files:**
- Delete:
  - `docker-compose.yaml`, `start-platform.sh`
  - `trino/`, `seaweedfs/`
  - `spark/conf/spark-defaults.conf`, `jupyter/spark-defaults.conf`
  - `tests/e2e/`
  - `hack/check-compose-parity.py`, `tests/unit/test_compose_parity.py`
  - `hive-metastore/hive-site.xml`, `hive-metastore/entrypoint.sh`
  - `.github/instructions/docker-compose.instructions.md`
- Modify:
  - `jupyter/Dockerfile` (drop the `COPY spark-defaults.conf`)
  - `.github/workflows/chart-lint.yaml` (drop the parity step)
  - `hack/workstation/e2e.sh` (use `charts/data-lab/files/e2e_lakehouse.py`)
  - `charts/data-lab/templates/tests/test-lakehouse-e2e.yaml` (comment: single copy)
  - `jupyter/notebooks/lab_utils.py` (endpoint constants)
  - every notebook's Compose branch or message
  - README.md, AGENTS.md, CLAUDE.md, docs/VERSIONS.md, docs/WORKSTATION_TESTING.md
- Test: `tests/unit/test_no_compose.py`, `tests/unit/test_lab_utils_endpoints.py`

**Interfaces:**
- Produces: `lab_utils.endpoint(name: str) -> str`, which returns `os.environ[name]` or raises
  `RuntimeError("<name> is not set: run this notebook in the lab's Jupyter (Helm chart charts/data-lab)")`.
  Module constants (`TRINO_HOST`, `REST_URI`, `S3_ENDPOINT`, `OLLAMA_URL`, `QDRANT_URL`, `PHOENIX_URL`, ...) become
  lazily read: either use `endpoint()` at the call site or keep `os.environ.get(name)` with no default. `TRINO_PORT`
  keeps its default `8080`.

- [ ] **Step 1: Write the failing tests**
  - `test_no_compose.py::test_compose_files_are_gone` asserts that none of these exist: `docker-compose.yaml`,
    `start-platform.sh`, `trino`, `seaweedfs`, `tests/e2e`, `hack/check-compose-parity.py`,
    `jupyter/spark-defaults.conf`, `spark/conf`.
  - `test_no_compose.py::test_living_docs_do_not_mention_compose` checks README.md, AGENTS.md, CLAUDE.md,
    `docs/WORKSTATION_TESTING.md`, `docs/VERSIONS.md` and every `jupyter/notebooks/**/*.ipynb` (case-insensitive). It
    asserts none contains `docker compose`, `docker-compose` or `start-platform`. One history line in VERSIONS §7 is
    allowed: `Compose removed in chart 0.4.0`.
  - `test_lab_utils_endpoints.py::test_missing_endpoint_env_raises_clear_error`: `monkeypatch.delenv("TRINO_HOST")`,
    then `endpoint("TRINO_HOST")` raises with `"run this notebook in the lab's Jupyter"`.
- [ ] **Step 2: Run them to see them fail.** Run `python3 -m pytest tests/unit/test_no_compose.py tests/unit/test_lab_utils_endpoints.py -q`. Expected: 3 failures.
- [ ] **Step 3: Delete the files, update the Dockerfile, CI, `e2e.sh` and `lab_utils`.** Notebooks lose their
  "Docker Compose: nothing to show" branches. `semantic_available()` and `require_ops()` messages drop the Compose
  wording. `dbt/lakehouse_demo/profiles.yml` keeps `env_var('TRINO_HOST')` with no `'trino'` default.
- [ ] **Step 4: Update the living docs.** README: remove the Compose quick start and make the chart path primary.
  AGENTS/CLAUDE: drop the Compose pitfalls (2, 4, 9's Compose parts) and the "Compose frozen" lines; renumber nothing,
  mark the removed pitfalls "removed in 0.4.0". VERSIONS: add the §7 history line. WORKSTATION_TESTING: drop step 2.
- [ ] **Step 5: Run everything.** Run `python3 -m pytest tests/chart tests/unit -q` and `helm lint` on each `ci/*.yaml`. Expected: all pass.
- [ ] **Step 6: Commit** `chore: remove Docker Compose; the Helm chart is the only way to run the lab`

### Task 2: Chart `llm` values, Jupyter env, Ollama context (chart 0.4.0)

**Files:**
- Modify:
  - `charts/data-lab/values.yaml` (new `llm` block per spec D2; `ollama.contextLength: 16384`)
  - `charts/data-lab/templates/_helpers.tpl` (`datalab.llmProvider`, `datalab.llmEnv`)
  - `charts/data-lab/templates/jupyter.yaml` (include `datalab.llmEnv`)
  - `charts/data-lab/templates/endpoints-configmap.yaml` (non-secret `LLM_*` keys)
  - `charts/data-lab/templates/ollama.yaml` (`OLLAMA_CONTEXT_LENGTH`)
  - `charts/data-lab/templates/validate.yaml`
  - `Chart.yaml` (0.4.0), `CHANGELOG.md`, schema, README (helm-docs)
- Test: `tests/chart/test_llm.py`

**Interfaces:**
- Produces Jupyter env (Task 4 reads these):
  - plain values: `LLM_PROVIDER`, `LLM_BASE_URL`, `LLM_CHAT_MODEL`, `LLM_TOOL_MODEL`, `LLM_EMBED_BASE_URL`,
    `LLM_EMBED_MODEL`;
  - from `secretKeyRef`: `LLM_API_KEY`, `LLM_EMBED_API_KEY`, only when a Secret is set.
- With `provider` resolving to `ollama`:
  - `LLM_BASE_URL` = `http://<fullname>-ollama:11434/v1`;
  - models default as in Global Constraints;
  - the embed URL equals the base URL.
- With `none`: only `LLM_PROVIDER=none`.

- [ ] **Step 1: Write the failing tests** in `tests/chart/test_llm.py` (use `render` and `env_of` from `conftest`):
  - `test_auto_is_none_without_ollama`: the default render's jupyter env has `LLM_PROVIDER == "none"` and no
    `LLM_API_KEY`.
  - `test_auto_is_ollama_with_ollama`: with `ollama.enabled=true`:
    - `LLM_BASE_URL == "http://r-data-lab-ollama:11434/v1"`
    - `LLM_TOOL_MODEL == "qwen3:4b-instruct"`
    - `LLM_EMBED_MODEL == "mxbai-embed-large:latest"`
  - `test_openai_mode`: with `llm: {provider: openai, baseUrl: https://openrouter.ai/api/v1, chatModel: a, toolModel: b, auth: {existingSecret: k}}`:
    - `LLM_API_KEY.valueFrom.secretKeyRef == {name: k, key: api-key}`
    - `LLM_EMBED_MODEL` is absent
  - `test_openai_requires_secret`: `provider: openai` without `auth.existingSecret` raises a `RenderError` matching
    `llm.auth.existingSecret`.
  - `test_openai_requires_models` and `test_ollama_mode_requires_ollama_enabled`: each raises a `RenderError`.
  - `test_ollama_context_length`: the ollama container env has `OLLAMA_CONTEXT_LENGTH == "16384"`.
  - `test_no_key_in_configmaps`: no rendered ConfigMap value contains the string `api-key` value or `LLM_API_KEY`.
- [ ] **Step 2: Run them to see them fail.** Run `python3 -m pytest tests/chart/test_llm.py -q`. Expected: failures (no `llm` values).
- [ ] **Step 3: Implement the values, helpers, templates and validation.** Bump the chart to 0.4.0, add the CHANGELOG
  entry, regenerate the schema and helm-docs.
- [ ] **Step 4: Run all tests and lint.** Run `python3 -m pytest tests/chart tests/unit -q` and lint on each profile. Expected: all pass; golden default unchanged except the new `LLM_PROVIDER=none` env (update the golden env assertion if one exists, not the resource list).
- [ ] **Step 5: Commit** `chart 0.4.0: llm block (ollama | openai | none) for Jupyter, Ollama context length`

### Task 3: Semantic server traces to Phoenix

**Files:**
- Modify:
  - `charts/data-lab/values.yaml` (`semantic.server.tracing.enabled: true`, `semantic.server.tracing.headersSecret: ""`)
  - `charts/data-lab/templates/_semantic.tpl` (`datalab.semanticServerConfig`: `observability.otlpEndpoint`)
  - `charts/data-lab/templates/semantic.yaml` (env `OTEL_EXPORTER_OTLP_HEADERS` from the headers Secret, key `otlp-headers`)
  - `charts/data-lab/templates/NOTES.txt`
  - schema, helm-docs, CHANGELOG (same 0.4.0 entry)
- Test: `tests/chart/test_semantic_tracing.py`

**Interfaces:**
- Produces the server config key `observability.otlpEndpoint: http://<fullname>-phoenix:6006/v1/traces` when
  semantic, Phoenix and `tracing.enabled` are on, and either `phoenix.auth.enabled` is false or a headers Secret is set.

- [ ] **Step 1: Write the failing tests:**
  - `test_tracing_to_phoenix`: with semantic + phoenix on (`api_versions=API` as in `test_semantic.py`),
    `server_config(docs)["observability"]["otlpEndpoint"] == "http://r-data-lab-phoenix:6006/v1/traces"`.
  - `test_no_tracing_when_phoenix_off`: semantic on, phoenix off → no `observability` key.
  - `test_auth_without_headers_secret_disables_tracing`: `phoenix.auth.enabled=true` and no headers Secret → no
    `observability` key, and NOTES contains `semantic.server.tracing.headersSecret`.
  - `test_headers_secret_env`: with `headersSecret: ph` → the server container env `OTEL_EXPORTER_OTLP_HEADERS`
    secretKeyRef `{name: ph, key: otlp-headers}`.
- [ ] **Step 2: Run them to see them fail.** Run `python3 -m pytest tests/chart/test_semantic_tracing.py -q`. Expected: 4 failures.
- [ ] **Step 3: Implement the template changes.** Regenerate the schema and helm-docs.
- [ ] **Step 4: Run all tests, lint the semantic profile, and push.** The kind `semantic` CI scenario must stay green
  (the server starts with an OTLP endpoint, and failed exports must not crash it). Run the tests and lint, push, and
  check the CI run.
- [ ] **Step 5: Commit** `chart: semantic server exports OTLP traces to Phoenix`

### Task 4: `lab_utils` on the OpenAI-compatible API

**Files:**
- Modify: `jupyter/notebooks/lab_utils.py`
- Test: `tests/unit/test_lab_utils_llm.py`; adapt `tests/unit/test_lab_utils_semantic.py` (fakes move from `ollama.Client` to an OpenAI-shaped client)

**Interfaces:**
- Produces:
  - `LLM_PROVIDER: str`, `TOOL_MODEL`, `CHAT_MODEL`, `EMBED_MODEL` (read from env; `TOOL_MODEL` falls back to
    `qwen3:4b-instruct` only when the provider is `ollama`)
  - `MAX_TOKENS = 1024`
  - `llm_available() -> bool` (False when the provider is `none` or unset)
  - `embeddings_available() -> bool`
  - `skip_unless_llm(section: str) -> bool`: prints `skipped (no LLM configured): <section>` and returns False when
    unavailable
  - `require_llm() -> None` (raises `RuntimeError` with the reason)
  - `openai_client() -> openai.OpenAI`
  - `_chat(model, messages, tools=None, max_tokens=MAX_TOKENS) -> dict`, which returns
    `{"message": {"content": str, "tool_calls": [{"function": {"name", "arguments": dict}}]}}`. That is the shape the
    existing loops consume. Arguments are JSON-decoded.
  - `mcp_tool_to_openai(tool) -> dict` (the renamed `mcp_tool_to_ollama`, same normalizer)
  - `require_tool_model(model=TOOL_MODEL) -> None`
  - `setup_tracing(project: str) -> bool`: registers Phoenix OTLP with the OpenInference OpenAI instrumentation, never
    raises, and returns whether it is on
  - `traced_tool_call(server: str, name: str, args: dict, call: Callable[[], dict]) -> dict`: wraps one MCP call in a
    span with `tool.name`, `tool.server`, `tool.arguments`, and for `query_metric` results `semantic.request_hash` and
    `semantic.sql`
- `ask_semantic` and `ask_raw_sql` keep their signatures and return shapes. `ask_ops` is removed in Task 7.

- [ ] **Step 1: Write the failing tests** in `test_lab_utils_llm.py` (with a fake `openai_client` via monkeypatch):
  - `test_chat_decodes_tool_call_arguments`: a fake returns a tool call with arguments `'{"model":"dbt_demo"}'` →
    `_chat(...)["message"]["tool_calls"][0]["function"]["arguments"] == {"model": "dbt_demo"}`.
  - `test_chat_caps_max_tokens`: the fake records kwargs, so `max_tokens == 1024`, and there is no `think` or `num_ctx`.
  - `test_chat_error_names_endpoint`: the fake raises `openai.APIStatusError(401)` → `RuntimeError` mentions
    `LLM_BASE_URL` and `401`.
  - `test_llm_available_modes`: `none` → False; `ollama` → True; `openai` → True.
  - `test_skip_unless_llm_prints`: with `none`, the captured stdout starts `skipped (no LLM configured):`.
  - `test_embeddings_available_false_without_model`: `openai` with no `LLM_EMBED_MODEL` → False.
  - `test_require_tool_model_probe_rejects_no_tools`: `openai`; the fake raises
    `BadRequestError("tools not supported")` on a probe with one dummy tool → `RuntimeError` naming the model.
  - `test_tracing_setup_never_raises`: `phoenix.otel.register` raises → `setup_tracing("x") is False`.
  - `test_traced_tool_call_tags_request_hash`: an in-memory span exporter shows `semantic.request_hash` from a fake
    `query_metric` result.
- [ ] **Step 2: Run them to see them fail.** Run `python3 -m pytest tests/unit/test_lab_utils_llm.py -q`.
- [ ] **Step 3: Implement.** Port `ask_semantic` and `ask_raw_sql` onto `_chat`, and route `ask_semantic`'s MCP calls
  through `traced_tool_call`. Adapt the existing semantic tests' fakes; their assertions stay.
- [ ] **Step 4: Run the tests.** Run `python3 -m pytest tests/chart tests/unit -q`. Expected: all pass.
- [ ] **Step 5: Commit** `lab_utils: one OpenAI-compatible client (ollama | openai | none), AI-off helpers, tool spans`

### Task 5: Jupyter image dependencies

**Files:**
- Modify: `jupyter/Dockerfile` (pip pins); docs/VERSIONS.md §8
- Test: `tests/unit/test_jupyter_image_pins.py`

- [ ] **Step 1: Write the failing test.** `test_ai_packages_pinned` parses `jupyter/Dockerfile` and asserts it contains:
  - `strands-agents[openai]==1.65.0`
  - `openinference-instrumentation-openai==`
  - `langchain-openai==`
  - `openai==`
  - and no unpinned `ollama` line (the `ollama` package is no longer imported by `lab_utils`)
- [ ] **Step 2: Run it to see it fail.**
- [ ] **Step 3: Pin the packages** to the current PyPI versions at implementation time, checked to resolve together with
  `pip install --dry-run` in a python:3.12 container. Remove `ollama` and `langchain-ollama` if nothing else imports
  them. Run `grep -r "import ollama\|langchain_ollama" jupyter/` to check.
- [ ] **Step 4: Build the image** with `docker build -t datalab-playground/jupyter:test jupyter` (or CI's build step).
  Expected: it builds, and `python -c "import strands, openai, langchain_openai"` passes inside it.
- [ ] **Step 5: Commit** `jupyter: openai, langchain-openai, Strands and OpenInference packages`

### Task 6: Notebooks on the new client, with AI-off paths and Phoenix sections

**Files:**
- Modify (regenerate from percent sources with the notebook builder used earlier, or edit the cells directly):
  - `data_lab_playground.ipynb` (Ollama, LangChain and AI-summary cells → `openai_client()` / `ChatOpenAI`, each
    gated by `skip_unless_llm`)
  - `rag_demo.ipynb` (`OpenAIEmbeddings(base_url=LLM_EMBED_BASE_URL, model=EMBED_MODEL, check_embedding_ctx_length=False)`,
    `ChatOpenAI`; sections 2–3 gated by `embeddings_available()` / `llm_available()`)
  - `semantic/01_chat_with_your_data.ipynb` (`setup_tracing("semantic-layer")`; the chat and score cells gated; a new
    "See it in Phoenix" markdown cell plus a code cell printing the request hashes of the last answer's tool calls)
- Test: `tests/unit/test_notebooks_ai_off.py`

- [ ] **Step 1: Write the failing test.** `test_ai_cells_are_gated` scans these notebooks' code cells. Every cell that
  calls `ChatOpenAI(`, `OpenAIEmbeddings(`, `ask_semantic(`, `ask_raw_sql(` or `openai_client(` must also contain
  `skip_unless_llm(`, `llm_available()` or `embeddings_available()`, or follow a `require_llm()` cell. Also:
  `test_no_ollama_imports` asserts no notebook imports `ollama` or `langchain_ollama`.
- [ ] **Step 2: Run it to see it fail.**
- [ ] **Step 3: Edit the notebooks.**
- [ ] **Step 4: Run the tests.** Then, in the sandbox, run the stack headless twice. That is the earlier smoke-stack
  approach: Spark Connect + HMS + Trino + SeaweedFS, with a CPU Ollama serving `/v1`:
  - with `LLM_PROVIDER=none`: every notebook finishes, and skipped sections print the line;
  - with `LLM_PROVIDER=ollama`: the answers match the earlier verified runs (semantic 6/6, the RAG answers name
    firmware 2.4.1 and the dead-on-arrival SSDs).
- [ ] **Step 5: Commit** `notebooks: OpenAI-compatible client, AI-off paths, semantic traces in Phoenix`

### Task 7: Strands agents for `ops/01` and `ops/02`

**Files:**
- Create:
  - `jupyter/notebooks/lab_agents.py` (Strands helpers; not named `strands*` to avoid shadowing the package)
  - `jupyter/notebooks/ops/02_spark_history_agent.ipynb` (workload and tools-by-hand per
    `docs/superpowers/plans/2026-11-16-spark-history-agent-notebook.md`; agent via `lab_agents`)
- Modify:
  - `jupyter/notebooks/ops/01_pipeline_agent.ipynb` (agent cells → `lab_agents.ops_agent(...)`)
  - `jupyter/notebooks/lab_utils.py` (remove `ask_ops`, `OPS_NUM_CTX`; keep `OPS_TOOLS`, `ops_servers`, `ops_call`,
    `require_ops`)
  - `hack/workstation/e2e.sh` (add `ops/02`)
- Test: `tests/unit/test_lab_agents.py`

**Interfaces:**
- Consumes: `lab_utils.OPS_TOOLS`, `ops_servers()`, `TOOL_MODEL`, `MAX_TOKENS`, `LLM_*` env, `setup_tracing`.
- Produces:
  - `lab_agents.strands_model() -> strands.models.openai.OpenAIModel`, built with
    `client_args={"base_url": LLM_BASE_URL, "api_key": LLM_API_KEY or "ollama"}`, `model_id=TOOL_MODEL` and
    `params={"max_tokens": MAX_TOKENS, "temperature": 0}`.
  - `lab_agents.mcp_clients(servers: list[str]) -> list[strands.tools.mcp.MCPClient]`, over streamable HTTP to
    `ops_servers()[s]`.
  - `lab_agents.ops_agent(servers: list[str] = ["airflow", "spark_history"]) -> contextmanager[strands.Agent]`. Its
    tools are each client's tools filtered to `OPS_TOOLS[server]`. The system prompt is the former `OPS_PROMPT` with
    today's date. Strands telemetry goes to Phoenix project `ops-agents`.
  - `lab_agents.ask(agent, question) -> dict` returns `{"answer": str, "steps": [{"tool", "args", "result"}]}`, with
    steps taken from the agent's message history.

- [ ] **Step 1: Write the failing tests** (Strands imported, MCP and model faked):
  - `test_strands_model_uses_llm_env`: env `LLM_BASE_URL=http://x/v1`, `LLM_TOOL_MODEL=m` → the model config has
    `model_id == "m"` and `client_args["base_url"] == "http://x/v1"`, plus `params["max_tokens"] == 1024`.
  - `test_ops_agent_offers_only_allowlisted_tools`: a fake MCP client lists `trigger_dag`, `list_dag_runs` and
    `get_task_logs` → the agent tool names are `{"list_dag_runs", "get_task_logs"}`.
  - `test_ops_agent_prompt_has_today`: today's ISO date is in `agent.system_prompt`.
  - `test_ask_collects_steps`: a scripted fake model (Strands' test pattern: a model yielding a tool use, then text) →
    `ask(...)["steps"][0]["tool"] == "list_dag_runs"`.
  - `test_ops_agent_skips_without_llm`: `LLM_PROVIDER=none` → `ops_agent` raises `RuntimeError` naming the provider
    (the notebook cell catches it with `skip_unless_llm`).
- [ ] **Step 2: Run them to see them fail.** Run `python3 -m pytest tests/unit/test_lab_agents.py -q` (install `strands-agents[openai]==1.65.0` in the test env; add it to chart-lint's pip line).
- [ ] **Step 3: Implement `lab_agents.py`, rewrite the `ops/01` agent cells and build `ops/02`.** `ops/02` agent
  questions:
  - "Why was my last query slow?"
  - "Which stage had the most skew?"
  - "How much less data did the filtered query read?"
- [ ] **Step 4: Run the tests, then run ops/01 and ops/02 headless in the sandbox** (Airflow + both MCP servers + CPU
  Ollama `/v1`, as in the earlier runs). Expected:
  - ops/01 names the failed or successful `northwind_daily` run and the longest Spark application;
  - ops/02's agent cites the skewed stage;
  - traces appear in Phoenix project `ops-agents`.
- [ ] **Step 5: Commit** `ops: Strands agents for ops/01 and the new ops/02 Spark History notebook`

### Task 8: No-GPU workstation profile and docs

**Files:**
- Create: `charts/data-lab/ci/workstation-nogpu-values.yaml` (workstation profile minus Ollama/GPU, `llm.provider: none`)
- Modify:
  - `hack/workstation/deploy.sh` and `hack/workstation/_common.sh` (`--profile gpu|nogpu`, default `gpu`)
  - `hack/workstation/build-images.sh` (no Ollama model pull with `nogpu`)
  - docs/WORKSTATION_TESTING.md (no-GPU section; the exact OpenRouter commands: `kubectl -n datalab create secret generic llm-key --from-literal=api-key=...` plus `--set llm.provider=openai,llm.baseUrl=https://openrouter.ai/api/v1,llm.chatModel=...,llm.toolModel=...,llm.auth.existingSecret=llm-key`)
  - README.md, AGENTS.md (pitfall 18a rewritten for the OpenAI-compatible path), CLAUDE.md, docs/TODO_MANUAL_TESTS.md
- Test: `tests/chart/test_workstation_profile.py` (extend)

- [ ] **Step 1: Write the failing tests:**
  - `test_nogpu_profile_has_no_ollama_and_llm_off`: rendering `workstation-nogpu-values.yaml` gives no ollama workload,
    and jupyter `LLM_PROVIDER == "none"`.
  - `test_nogpu_profile_with_openrouter`: the same profile plus the openai `--set` values renders `LLM_API_KEY` from
    `llm-key`.
- [ ] **Step 2: Run them to see them fail.**
- [ ] **Step 3: Create the profile, script flags and docs.** TODO_MANUAL_TESTS gains the maintainer checks:
  - GPU `e2e.sh`;
  - a no-GPU deploy with an OpenRouter key: all notebooks run, with real answers;
  - a no-GPU deploy with AI off: all notebooks finish with skip lines;
  - Phoenix shows `semantic.query` spans in `default`, and notebook spans in `semantic-layer` / `ops-agents`.
- [ ] **Step 4: Run all tests and lint on every profile, push, and check CI (chart-lint plus the four kind scenarios).**
- [ ] **Step 5: Commit** `workstation: no-GPU profile (bring your own OpenAI-compatible endpoint or AI off)`
