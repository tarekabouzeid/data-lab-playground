# Slimmer lab: no Compose, LLMs from anywhere, semantic traces, Strands agents

Status: **draft for review** (2026-11-17). Nothing is built until the maintainer approves this document and the plan
(`docs/superpowers/plans/2026-11-17-slim-lab-llm-providers.md`).

## Intent and why

The lab grew to about 15 notebooks and 33 chart toggles, and it needs a GPU workstation to run the AI parts. The
maintainer wants it smaller and easier for others to run:

1. **Remove Docker Compose.** It is frozen, cannot run the semantic, Airflow or ops parts, and doubles the maintenance
   (two configs, a parity checker, two e2e paths). The Helm chart becomes the only way to run the lab.
2. **No GPU required.** Users either bring any OpenAI-compatible endpoint (OpenRouter, a vLLM box, a hosted provider)
   with a base URL, model names and a key, or turn the AI parts off entirely. With AI off, every notebook still runs
   and says plainly which sections it skipped.
3. **Semantic layer traces in Phoenix.** Phoenix is deployed but only the RAG notebook uses it. The semantic layer's
   own spans and the LLM calls that drive it should be visible there.
4. **Spark agent on AWS Strands.** The planned `ops/02` Spark History agent notebook is built with the Strands Agents SDK
   instead of the hand-written tool loop.

Success: a user without a GPU installs the chart, points it at OpenRouter (or sets AI off), and every notebook runs to
the end. The repository has no Compose files, and Phoenix shows the semantic layer's queries.

## Goals

- One deployment path (the chart). Delete Compose-only files, tests, CI steps and documentation.
- One LLM client path in `lab_utils`: the OpenAI-compatible API, which local Ollama also serves (`/v1`). Three modes:
  - `ollama`: the chart's own Ollama, with or without a GPU.
  - `openai`: any OpenAI-compatible endpoint plus a key from a Secret.
  - `none`: AI off.
- Notebooks branch on one helper (`lab_utils.llm_available()`) and print what they skip.
- The semantic server exports OTLP traces to Phoenix. The notebooks' LLM and tool calls are traced too, tagged with the
  server's `request_hash` so the two can be matched.
- `ops/02_spark_history_agent.ipynb` uses Strands, with the same model settings and Phoenix tracing.

## Non-goals

- No new services: no LiteLLM proxy, no OpenTelemetry collector, no Bedrock.
- No change to the Sandpiper defaults (the golden default render stays the 0.1.0 resource set).
- No upstream change to the Semantic Operator. Its server does not continue the caller's trace (no `traceparent`
  extraction); we do not patch it.
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
- Semantic server (kubeflow/semantic-operator@4fd7057):
  - config key `observability.otlpEndpoint` (OTLP/HTTP, empty means off);
  - spans named `semantic.query` with the model, model version, adapter and request hash;
  - service name `semantic-server`;
  - no trace-context propagation.

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
  embeddings:
    baseUrl: ""         # default: same as baseUrl (ollama: the chart's Ollama)
    model: ""           # default: mxbai-embed-large:latest with ollama; required for rag_demo with openai
    auth:
      existingSecret: ""   # default: llm.auth
      apiKeyKey: api-key
```

- Jupyter gets `LLM_PROVIDER`, `LLM_BASE_URL`, `LLM_CHAT_MODEL`, `LLM_TOOL_MODEL`, `LLM_EMBED_BASE_URL`,
  `LLM_EMBED_MODEL`, and `LLM_API_KEY` / `LLM_EMBED_API_KEY` from `secretKeyRef`. The endpoints ConfigMap carries the
  non-secret keys.
- Validation (`validate.yaml`): `provider: openai` requires `baseUrl`, `auth.existingSecret`, `chatModel` and
  `toolModel`. `provider: ollama` requires `ollama.enabled`.
- Embeddings are separate because not every chat provider serves embeddings (OpenRouter is chat-first). Without an
  embedding model, `rag_demo` stops early with that reason.
- `ollama.contextLength` (new, default 16384) sets `OLLAMA_CONTEXT_LENGTH`. The OpenAI-compatible API has no
  per-request `num_ctx`, so the context must be raised server-side for the ops agents (15k tokens of tool schemas
  measured).

### D3. One client path in `lab_utils`

- `_chat()` uses the `openai` SDK (`base_url=LLM_BASE_URL`, `api_key=LLM_API_KEY or "ollama"`) with
  `chat.completions.create(..., tools=..., max_tokens=...)`. The `ollama` package is no longer used.
- These stay:
  - `mcp_tool_to_ollama` → renamed `mcp_tool_to_openai`, with the schema normalizer kept (harmless elsewhere);
  - the model-name guard and the grain guard;
  - the curated ops tool set and today's date in the ops prompt.
- Dropped: `think=False` and `num_ctx` (Ollama-only options). The instruct tool model does not think; the context is
  set server-side (D2).
- `require_tool_model()`:
  - with `ollama`, keeps the `/api/show` capability check;
  - with `openai`, cannot ask, so it does one tiny tool-call probe and reports a clear error if tools are unsupported.
- `llm_available()` returns False for `none`. `require_llm()` raises a clear message for whole-notebook AI pages.
- LangChain cells (`rag_demo`, the playground) move from `langchain_ollama` to `langchain_openai`.
  `OpenAIEmbeddings(check_embedding_ctx_length=False)` is needed for non-OpenAI servers.

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

### D6. Semantic layer traces in Phoenix

- **Server side.** When `semantic.enabled` and `phoenix.enabled`, the chart renders `observability.otlpEndpoint:
  http://<phoenix>:6006/v1/traces` into the semantic server config, behind a new value `semantic.server.tracing.enabled`
  (default true; inert unless semantic is on, so the default render is unchanged).
  - Phoenix with auth on needs an API key, which Phoenix issues in its UI; the chart's Phoenix Secret only holds the
    signing secret and admin password. So `semantic.server.tracing.headersSecret` (new, empty by default) names a Secret
    whose key `otlp-headers` (e.g. `authorization=Bearer <key>`) becomes `OTEL_EXPORTER_OTLP_HEADERS`, which
    `otlptracehttp` honours. With `phoenix.auth.enabled` and no headers Secret, tracing stays off and NOTES.txt says why.
  - Server spans land in Phoenix's `default` project: the server's resource only sets `service.name`, and we do not
    patch upstream.
- **Notebook side.**
  - `lab_utils` registers Phoenix tracing (project `semantic-layer`) with the OpenInference OpenAI instrumentation, so
    every `_chat` call is a traced LLM span.
  - Each MCP tool call gets its own span with `tool.name`, arguments and, for `query_metric`, `semantic.request_hash` and
    the compiled SQL.
- **Linking.** The notebook and server spans are separate traces. They are matched by `semantic.request_hash`, which is
  on both. `semantic/01` gets a short "See it in Phoenix" section that shows how.

### D7. Strands for the ops agents

- `ops/02_spark_history_agent.ipynb` follows the existing plan
  (`docs/superpowers/plans/2026-11-16-spark-history-agent-notebook.md`) for the workload and the tools-by-hand part.
  Its agent is a Strands `Agent`:
  - model: `strands.models.openai.OpenAIModel(client_args={base_url, api_key}, model_id=LLM_TOOL_MODEL, params={max_tokens})`;
  - tools: an `MCPClient` over streamable HTTP to the Spark History MCP, filtered to the read-only allowlist.
- Tracing: Strands' built-in OpenTelemetry, exported to Phoenix (project `ops-agents`).
- **My call, open for review:** `ops/01` moves to Strands too, with Airflow plus Spark History MCP clients, so both ops
  agents share one framework and `lab_utils.ask_ops` is retired.
  - `semantic/01` keeps its explicit loop, because that notebook's purpose is to show each step (prompt → tool call →
    compiled SQL → answer), and its guards (model name, grain) live in that loop.
  - The alternative is Strands everywhere, at the cost of rewriting those guards as Strands hooks and hiding the loop
    the notebook teaches.
- Pinned packages in the Jupyter image:
  - `strands-agents[openai]==1.65.0`
  - `openinference-instrumentation-openai`
  - `langchain-openai`
  - Strands' own OTel export needs no extra package.

### D8. Versions

- Chart `0.4.0`: new values (`llm`, `ollama.contextLength`, `semantic.server.tracing`) plus the no-GPU profile. Additive.
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

For the agent framework, Strands was the maintainer's choice. The open part is its scope (D7).

## Left to the builder

- Exact skip messages and cell layout in the notebooks.
- How the tool allowlist is shared between `lab_utils` and the Strands notebooks (one constant).
- Whether the `require_tool_model` probe for `openai` caches its result per session.

## Verification

- Unit and chart tests:
  - LLM modes (render, env, validation);
  - the `_chat` contract with a fake OpenAI client;
  - skip helpers;
  - the server tracing config;
  - Compose files gone;
  - one e2e script copy.
- kind CI (the four scenarios, unchanged).
- Sandbox: notebooks headless with CPU Ollama through `/v1` (mode `ollama`) and with `none`. Mode `openai` needs a real
  key, so it goes to the maintainer.
- Maintainer (TODO): GPU workstation `e2e.sh`, a no-GPU profile with an OpenRouter key, and Phoenix traces checked in the
  UI.
