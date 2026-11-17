# Slimmer lab: no Compose, LLMs from anywhere, Strands agents traced in Phoenix

Status: **draft for review** (2026-11-17, revised after the maintainer's first review). Nothing is built until the maintainer approves this document and the plan
(`docs/superpowers/plans/2026-11-17-slim-lab-llm-providers.md`).

## Intent and why

The lab grew to about 15 notebooks and 33 chart toggles, and it needs a GPU workstation to run the AI parts. The
maintainer wants it smaller and easier for others to run:

1. **Remove Docker Compose.** It is frozen, cannot run the semantic, Airflow or ops parts, and doubles the maintenance
   (two configs, a parity checker, two e2e paths). The Helm chart becomes the only way to run the lab.
2. **No GPU required.** Users either bring any OpenAI-compatible endpoint (OpenRouter, a vLLM box, a hosted provider)
   with a base URL, model names and a key, or turn the AI parts off entirely. With AI off, every notebook still runs
   and says plainly which sections it skipped.
3. **Chat with your data on AWS Strands, traced in Phoenix.** `semantic/01`'s chat-with-your-data agent (the LLM using
   the Semantic Operator's MCP tools) is rebuilt with the Strands Agents SDK, and that agent's traces are what goes to
   Phoenix. Nothing else is added to Phoenix (maintainer, 2026-11-17).
4. **Spark agent on AWS Strands.** The planned `ops/02` Spark History agent notebook is built with Strands instead of
   the hand-written tool loop.

Success: a user without a GPU installs the chart, points it at OpenRouter (or sets AI off), and every notebook runs to
the end. The repository has no Compose files, and Phoenix shows each chat-with-your-data question as one trace: the
model's turns, the semantic-layer tool calls and the answer.

## Goals

- One deployment path (the chart). Delete Compose-only files, tests, CI steps and documentation.
- One LLM client path in `lab_utils`: the OpenAI-compatible API, which local Ollama also serves (`/v1`). Three modes:
  - `ollama`: the chart's own Ollama, with or without a GPU.
  - `openai`: any OpenAI-compatible endpoint plus a key from a Secret.
  - `none`: AI off.
- Notebooks branch on one helper (`lab_utils.llm_available()`) and print what they skip.
- `semantic/01`'s chat agent runs on Strands, and its traces (Strands' built-in OpenTelemetry) go to Phoenix project
  `semantic-layer`. Nothing else is newly sent to Phoenix.
- `ops/02_spark_history_agent.ipynb` uses Strands, with the same model settings.

## Non-goals

- No new services: no LiteLLM proxy, no OpenTelemetry collector, no Bedrock.
- No change to the Sandpiper defaults (the golden default render stays the 0.1.0 resource set).
- No upstream change to the Semantic Operator, and no server-side tracing: the server's own OTLP exporter stays off
  (maintainer: only the Strands chat agent goes to Phoenix).
- `RELEASE_NOTES.md` stays untouched until the maintainer decides the release.

## Anti-goals (failure even if it "works")

- A key ends up in values, a ConfigMap, a notebook or a log line. Keys come only from a Secret the user creates
  (`existingSecret`), as for every other credential in the chart.
- AI-off notebooks fail headless. They must finish, with clearly skipped cells.
- The notebooks fork into an "Ollama version" and an "API version". There is one code path.
- Removing Compose breaks the chart images. The `hive-metastore/`, `spark/` and `jupyter/` Dockerfiles stay; they build
  the chart's images.

## Constraints (from the repository)

- `values.yaml` is a public API: add keys only, each with a `# --` comment. Regenerate the schema and helm-docs. Bump the
  chart version and add a CHANGELOG entry.
- Notebook rules:
  - no hardcoded hosts (enforced by a test);
  - no `.master()`, `SparkContext` or RDDs;
  - no `DROP SCHEMA`;
  - no module named `dbt_*`.
- Ollama facts learned in real runs (AGENTS.md pitfall 18a):
  - Ollama rejects some valid tool schemas: keep the normalizer.
  - Its default context is 4,096 tokens.
  - Plain `qwen3:4b` is thinking-only; the tool model is `qwen3:4b-instruct`.
  - Cap output tokens on every call.
- Jupyter image: Python 3.12. Strands 1.65.0 needs Python ≥ 3.10 and supports OpenAI-compatible models (`openai`
  extra) and MCP servers over streamable HTTP.
- Semantic server (kubeflow/semantic-operator@4fd7057): MCP tools `list_models`, `list_metrics`, `list_dimensions`,
  `query_metric`; identity headers `X-Semantic-User` / `X-Semantic-Role` (header auth). It has its own OTLP exporter
  (`observability.otlpEndpoint`), which this design leaves off.
- OpenRouter serves embeddings as well as chat (maintainer, 2026-11-17), through the same OpenAI-compatible API.

## Decisions

### D1. Compose removal: what goes, what stays

| Goes | Stays |
|---|---|
| `docker-compose.yaml`, `start-platform.sh` | `hive-metastore/`, `spark/`, `jupyter/`, `airflow/`, `airflow-mcp/` Dockerfiles (chart images) |
| `trino/` (Compose image + catalogs; the chart uses stock `trinodb/trino` with a ConfigMap) | `charts/data-lab/files/e2e_lakehouse.py`, the single copy of the 19-check e2e (the helm test, the SparkApplication and `hack/workstation/e2e.sh` use it) |
| `seaweedfs/s3.json` (the chart generates keys) | `hive-metastore/*.xml.template` (baked into the HMS image) |
| `spark/conf/spark-defaults.conf`, `jupyter/spark-defaults.conf` and its `COPY` in `jupyter/Dockerfile` (classic Spark master) | `tests/chart`, `tests/unit` |
| `tests/e2e/` (`run-e2e.sh` and the duplicate `e2e_lakehouse.py`) | |
| `hack/check-compose-parity.py`, `tests/unit/test_compose_parity.py`, the parity step in CI | |
| `hive-metastore/hive-site.xml`, `hive-metastore/entrypoint.sh` (marked legacy/unused) | |
| `.github/instructions/docker-compose.instructions.md` | |
| Compose sections in README, AGENTS, CLAUDE, VERSIONS, WORKSTATION_TESTING and the notebooks | |

`lab_utils` endpoint constants lose their Compose defaults. When an endpoint variable is missing, the helper raises
"run this notebook in the lab's Jupyter (Helm chart)" instead of silently trying a Compose host name. The endpoints test
still forbids hardcoded hosts.

Historical docs (`docs/UPGRADE_PLAN.md`, `docs/K8S_HELM_PLAN.md`, `RELEASE_NOTES.md`, earlier plans) keep their Compose
mentions as history. Only living docs change.

### D2. LLM settings: one `llm` block, three modes

New top-level chart values (additive):

```yaml
llm:
  provider: ""          # "" = auto: "ollama" when ollama.enabled, else "none"; or "ollama", "openai", "none"
  baseUrl: ""           # openai mode: e.g. https://openrouter.ai/api/v1
  auth:
    existingSecret: ""  # openai mode: Secret holding the key (never in values)
    apiKeyKey: api-key
  chatModel: ""         # default: gemma3:4b with ollama; required with openai
  toolModel: ""         # default: qwen3:4b-instruct with ollama; required with openai (must support tool calling)
  embeddingModel: ""    # default: mxbai-embed-large:latest with ollama; required for rag_demo with openai
```

- Jupyter gets `LLM_PROVIDER`, `LLM_BASE_URL`, `LLM_CHAT_MODEL`, `LLM_TOOL_MODEL`, `LLM_EMBED_MODEL`, and
  `LLM_API_KEY` from `secretKeyRef`. The endpoints ConfigMap carries the non-secret keys.
- Validation (`validate.yaml`): `provider: openai` requires `baseUrl`, `auth.existingSecret`, `chatModel` and
  `toolModel`. `provider: ollama` requires `ollama.enabled`.
- Embeddings use the same endpoint and key as chat: OpenRouter, Ollama and most OpenAI-compatible providers serve both.
  Only the model name is separate. Without an embedding model set, `rag_demo` sections 2–3 print the reason and skip.
- `ollama.contextLength` (new, default 16384) sets `OLLAMA_CONTEXT_LENGTH`. The OpenAI-compatible API has no
  per-request `num_ctx`, so the context must be raised server-side for the ops agents (15k tokens of tool schemas
  measured).

### D3. One client path in `lab_utils`

- `_chat()` uses the `openai` SDK (`base_url=LLM_BASE_URL`, `api_key=LLM_API_KEY or "ollama"`) with
  `chat.completions.create(..., tools=..., max_tokens=...)`. The `ollama` package is no longer used.
- These stay:
  - `mcp_tool_to_ollama` → renamed `mcp_tool_to_openai`, with the schema normalizer kept (harmless elsewhere);
  - the curated ops tool set and today's date in the ops prompt (moved into the Strands agents, D7).
- `ask_semantic` and `ask_ops` (the hand-written tool loops) are retired; the agents move to Strands (D7).
  `ask_raw_sql` (one plain chat call, no tools) stays on `_chat`.
- Dropped: `think=False` and `num_ctx` (Ollama-only options). The instruct tool model does not think; the context is
  set server-side (D2).
- `require_tool_model()`:
  - with `ollama`, keeps the `/api/show` capability check;
  - with `openai`, cannot ask, so it does one tiny tool-call probe and reports a clear error if tools are unsupported.
- `llm_available()` returns False for `none`. `require_llm()` raises a clear message for whole-notebook AI pages.
- LangChain cells (`rag_demo`, the playground) move from `langchain_ollama` to `langchain_openai`.
  `OpenAIEmbeddings(check_embedding_ctx_length=False)` is needed for non-OpenAI servers. Their existing Phoenix
  instrumentation (`rag_demo`'s `northwind-rag` project, the playground's `ai-platform-demo`) is left as it is: see
  open question Q1.

### D4. AI-off behavior per notebook

| Notebook | AI off (`none`) |
|---|---|
| `data_lab_playground` | Runs; the Ollama, LangChain and AI-summary cells print "skipped: no LLM configured". |
| `rag_demo` | Section 1 (the numbers) runs; sections 2–3 print skipped. |
| `semantic/01` | The tools-by-hand, compiled-SQL and governance cells run (no LLM needed). The step-by-step chat and score table are skipped. |
| `ops/01`, `ops/02` | The tools-by-hand cells run; the agent cells are skipped. |
| iceberg, dbt | Unaffected. |

`hack/workstation/e2e.sh` keeps running every notebook. Skipped cells still let a notebook finish.

### D5. No-GPU workstation profile

- New `charts/data-lab/ci/workstation-nogpu-values.yaml`: Ollama off, `llm.provider: none`, the rest as the workstation
  profile.
- For OpenRouter, the user adds three `--set` values plus a Secret. `docs/WORKSTATION_TESTING.md` shows the exact
  commands, e.g. `kubectl create secret generic llm-key --from-literal=api-key=...`.
- `hack/workstation/setup-minikube.sh --gpu-mode none` already exists. `deploy.sh` gets `--profile gpu|nogpu`
  (default `gpu`).
- Running Ollama on CPU stays possible (`ollama.gpu.enabled=false`) but is not a profile. It was measured at minutes per
  agent turn, too slow to recommend.

### D6. Chat with your data on Strands, traced in Phoenix

- `semantic/01`'s agent is a Strands `Agent`:
  - model: the shared `OpenAIModel` (D7);
  - tools: an `MCPClient` over streamable HTTP to `SEMANTIC_MCP_URL`, sending `X-Semantic-User` / `X-Semantic-Role` for
    the chosen role (analyst, marketing, dach_sales);
  - system prompt: today's `SEMANTIC_PROMPT`, plus the published model's name, fetched once before the agent starts.
- The two guards from real runs become Strands hooks:
  - a `BeforeToolCallEvent` hook replaces an unknown `model` argument with the published model when exactly one is
    published;
  - the same hook drops `grain` when no requested dimension is a time dimension.
  Each correction is recorded on the step, so the notebook still shows it.
- The notebook keeps its step-by-step teaching. After each question it prints every tool call, its arguments and the
  server's result (compiled SQL and rows), read from the agent's message history. The score table (semantic vs raw
  text-to-SQL) stays.
- **Phoenix:** Strands' built-in OpenTelemetry tracer is exported over OTLP/HTTP to Phoenix
  (`PHOENIX_OTLP_HTTP_ENDPOINT`, project `semantic-layer`). One question becomes one trace: an agent span, model-call
  spans and a span per tool call. Nothing else is added: no server-side export, no extra spans in `lab_utils`, no other
  notebook's tracing changes.
  - With Phoenix auth on, Phoenix needs an API key, issued in its UI. The notebook reads it from `PHOENIX_API_KEY` if
    set and otherwise traces nothing, printing why. Tracing must never fail a cell.
- `semantic/01` adds a short "See it in Phoenix" section: open the `semantic-layer` project and find the trace whose
  tool span carries the compiled SQL shown above.

### D7. Strands everywhere an agent calls tools

- One helper module, `jupyter/notebooks/lab_agents.py`, builds:
  - the model: `strands.models.openai.OpenAIModel(client_args={base_url: LLM_BASE_URL, api_key}, model_id=LLM_TOOL_MODEL,
    params={max_tokens, temperature: 0})`;
  - MCP clients for the semantic, Airflow and Spark History servers (streamable HTTP), with tools filtered to read-only
    allowlists;
  - the agents: `semantic_agent(role)` (D6), `ops_agent(servers)` (`ops/01` with Airflow + Spark History, `ops/02` with
    Spark History only);
  - `ask(agent, question) -> {answer, steps}`.
- `ops/02_spark_history_agent.ipynb` follows the existing plan
  (`docs/superpowers/plans/2026-11-16-spark-history-agent-notebook.md`) for the workload and the tools-by-hand part.
- `ops/01` moves to Strands too, so every tool-calling agent in the lab uses one framework.
- Tracing for the ops agents: see open question Q1. The design sends them **no** traces until the maintainer answers.
- Pinned packages in the Jupyter image:
  - `strands-agents[openai]==1.65.0`
  - `langchain-openai`
  - the OTLP exporter `opentelemetry-exporter-otlp-proto-http` (Strands' tracer exports through it; check whether it is
    already a dependency).
  - `openinference-instrumentation-openai` is not needed.

### D8. Versions

- Chart `0.4.0`: new values (`llm`, `ollama.contextLength`) plus the no-GPU profile. Additive.
- The Jupyter image changes (new packages, no Spark config file). Custom image tags equal `appVersion`; whether this is
  `3.0.0` → `3.1.0` or `4.0.0` (Compose removal is breaking for Compose users) is **the maintainer's release decision**.
  The plan keeps `3.0.0` until then.

## Approaches considered

1. **One OpenAI-compatible client for everything (chosen).** Ollama, OpenRouter and vLLM all speak it. One code path, one
   set of tests; Strands and LangChain both have OpenAI-compatible adapters. Cost: lose Ollama-only knobs (`think`,
   per-request `num_ctx`), so the context moves server-side.
2. **Keep the Ollama client and add an OpenAI branch.** Lowest churn in `lab_utils`, but every helper, test and notebook
   has two paths, which is exactly the fork the anti-goals rule out.
3. **A LiteLLM proxy in the chart.** Uniform API with provider routing, but it adds a service, the opposite of making the
   lab smaller.

For the agent framework, Strands was the maintainer's choice, for the chat-with-your-data agent and the ops agents.

## Left to the builder

- Exact skip messages and cell layout in the notebooks.
- How the tool allowlists are shared between `lab_utils` (tools by hand) and `lab_agents` (one constant per server).
- Whether the `require_tool_model` probe for `openai` caches its result per session.

## Verification

- Unit and chart tests:
  - LLM modes (render, env, validation);
  - the `_chat` contract with a fake OpenAI client;
  - skip helpers;
  - the Strands guard hooks (model name, grain) and the step extraction;
  - Compose files gone;
  - one e2e script copy.
- kind CI (the four scenarios, unchanged).
- Sandbox: notebooks headless with CPU Ollama through `/v1` (mode `ollama`) and with `none`. Mode `openai` needs a real
  key, so it goes to the maintainer.
- Maintainer (TODO): GPU workstation `e2e.sh`, a no-GPU profile with an OpenRouter key, and Phoenix traces checked in the
  UI.

## Open questions for the maintainer

- **Q1. Phoenix scope.** "Only the Strands chat-with-your-data agent goes to Phoenix": should the ops agents (`ops/01`,
  `ops/02`, also Strands) send **no** traces, and should `rag_demo` and the playground keep their **existing** Phoenix
  tracing (projects `northwind-rag`, `ai-platform-demo`), or drop it too? This design assumes: ops agents untraced,
  existing tracing kept.
