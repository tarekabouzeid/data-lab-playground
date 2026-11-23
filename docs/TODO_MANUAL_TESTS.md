# TODO: what is left (manual, GPU workstation)

Everything else is done and in PR #7. These need a GPU machine with `ollama pull` access. Each item has a pass
criterion. Full checklist: [WORKSTATION_TESTING.md](WORKSTATION_TESTING.md).

- [ ] **Workstation end to end with the real models.**
  ```bash
  hack/workstation/setup-minikube.sh --gpu-mode device-plugin
  hack/workstation/build-images.sh        # also builds the lab Airflow image (airflow/Dockerfile)
  hack/workstation/deploy.sh
  hack/workstation/e2e.sh
  ```
  Pass: `workstation e2e passed`, including `northwind_daily run e2e-<n>: success` and
  `ok ops/01_pipeline_agent.ipynb`.
- [x] **`semantic/01` with `qwen3:4b-instruct`** (Qwen3-4B-Instruct-2507, the sandbox's stand-in; plain `qwen3:4b` is
  thinking-only and never reaches a tool call). Pass: the semantic layer scores 6/6, and the step-by-step cell runs `list_metrics → list_dimensions → query_metric`.
- [x] **`ops/01` with `qwen3:4b-instruct`.** Pass: it names the latest `northwind_daily` run and its task states, and
  Spark application ids from the History Server.
- [x] **`rag_demo` on GPU.** Pass: `numbers from: semantic layer (certified metrics)`; the router answer names firmware
  2.4.1 and the SSD answer the dead-on-arrival drives.
- [x] **Airflow UI** (`http://airflow.datalab.localhost:8080`): `northwind_daily` is unpaused. The latest run shows
  `build` logging `Done. PASS=50` and `published` succeeding.
- [x] **Confirm CI on the PR head is green**, including the kind `parallel` job: chart-lint and chart-e2e (all four kind
  scenarios) passed on 0de20f5 (runs 43975253629, 43975253647).
- [x] **Release:** v4.0.0 (Compose removed), `RELEASE_NOTES.md` entry written 2026-11-23. Tag `v4.0.0` after the merge to `main`.

## New in chart 0.4.0 (needs you; the sandbox has no cluster or LLM)

- [x] **GPU workstation**: `hack/workstation/deploy.sh && hack/workstation/e2e.sh`. Pass: `workstation e2e passed`,
  including `ok ops/01_pipeline_agent.ipynb` and `ok ops/02_spark_history_agent.ipynb`. `semantic/01` still scores 6/6
  with the Strands agent.
- [x] **No GPU, AI off** (2026-11-23: every e2e step passed except `rag_demo`, whose unguarded search cell is fixed in a5a302c; rerun passes): `deploy.sh --profile nogpu`, then `e2e.sh`. Pass: every notebook finishes with `skipped (no LLM configured)` lines.
- [x] **Your own OpenAI-compatible endpoint** (`llm.provider=openai`; see below): see `WORKSTATION_TESTING.md` §4b (chat + tool model, embeddings from the same endpoint and,
  once, from a different one via `llm.embeddingBaseUrl`). Pass: real answers in `rag_demo`, `semantic/01`, `ops/01`.
- [x] **Phoenix**: project `semantic-layer` shows one trace per `semantic/01` question with tool spans (the Strands spans go
  through `openinference-instrumentation-strands-agents`; if they render as plain spans, tell me) and no other new project.
- [x] **`ops/02`**: written from the Spark History MCP source (tool names) without a live server; the first run may need
  small fixes to the result parsing (`items()`) and to the `compare_sql_executions` argument names. Fixed in dbfe2e8.

## Next (development)

- [x] **`ops/02_spark_history_agent.ipynb`** (built in 0.4.0, see above), a notebook for the Spark History MCP server. It inspects the running Spark
  Connect session (`live`) and finished applications (`history`) by hand, then lets the tool model diagnose a skewed join
  and a missing partition filter. Plan: [2026-11-16-spark-history-agent-notebook.md](superpowers/plans/2026-11-16-spark-history-agent-notebook.md).

## Already verified

- 2026-11-22, workstation lab (minikube, RTX 5070 Ti, chart 0.4.1, `qwen3:4b-instruct` on Ollama `/v1`):
  - `hack/workstation/e2e.sh` in one pass: `workstation e2e passed`. That covers `helm test` and the SparkApplication
    (19/19), all 11 notebooks, `northwind_daily` (`success`), `ok ops/01_pipeline_agent.ipynb`,
    `ok ops/02_spark_history_agent.ipynb`, and 10/10 gateway hosts. The lab peaked at 17.1 GiB.
  - `semantic/01`: 6/6 with the Strands agent (raw text-to-SQL 1/6), governance unchanged. `rag_demo`: firmware 2.4.1 and dead-on-arrival SSDs.
  - Phoenix: `semantic-layer` has one trace per question with agent, LLM and tool spans. `northwind-rag` and
    `ai-platform-demo` still get traces (kept by design, open question Q1 in the 0.4.0 spec).
  - `ops/02`: the skewed join stage is 72.5× the median records and the filtered scan reads 33% of the bytes; the
    agent names the stage and the byte counts.
  - Fixed on the way: `helm test` failed on every run after the first (chart 0.4.1), `ops/02` against the live
    server, `test_no_compose` with Jupyter checkpoints.
- 2026-11-22, `llm.provider=openai` against a self-hosted OpenAI-compatible server on the host (reached at
  `http://host.minikube.internal:<port>/v1`; a reasoning chat model also used for tools, a separate embeddings model,
  key in an existing Secret, Ollama still deployed):
  - `semantic/01` 6/6 (raw 1/6), governance unchanged; `rag_demo` (firmware 2.4.1, dead SSDs); the tour; `ops/01`;
    `ops/02` (exact input bytes, a good explanation of the slower join).
  - The ops agents need >= 16k context: at the server's first 8k they were refused (11.3k / 9.7k tokens).
  - Embeddings from a different endpoint (`llm.embeddingBaseUrl` = the chart's Ollama, `mxbai-embed-large`): `rag_demo` passes.
  - Fixed on the way: agents fit a strict-context endpoint (tool result cap, the overflow wording of LiteLLM and
    llama.cpp), and Jupyter
    restarts when the endpoints ConfigMap changes (chart 0.4.2).
- 2026-11-22, chart 0.4.2 back on Ollama: `hack/workstation/e2e.sh` passed in one pass again (peak lab 14.2 GiB).

- 2026-11-16, workstation lab (minikube, RTX 5070 Ti, chart 0.3.1), each run from the browser:
  - `semantic/01`: 6/6 (raw text-to-SQL 1/6), the step-by-step cell runs `list_metrics → list_dimensions → query_metric`, `marketing` is denied the e-mail filter, `dach_sales` sees only DACH.
  - `ops/01`: it names the latest `northwind_daily` run, its task states and the Spark application id.
  - `rag_demo`: the router answer names firmware 2.4.1, the SSD answer the dead-on-arrival drives.
  - The tour notebook runs end to end.
  - A UI-triggered `northwind_daily` run succeeded with all 5 tasks, and `build` logged `Done. PASS=50`. Before that, two catch-up runs started together and collided in `dbt seed`; that is fixed by `max_active_runs=1`.
  - The lab peaked at 12.9 GiB.

- CI: chart-lint and the kind minimal, default and semantic jobs passed on 4b494d3. The semantic job also runs the DAG's
  `wait_for_published` against the real Semantic Operator.
- Every notebook ran headless in the sandbox, in Compose mode and in chart (Spark Connect) mode, on CPU with stand-in
  GGUF models from Docker Hub. Details: `docs/superpowers/plans/2026-10-30-northwind-data-story.md`.
- `northwind_daily` passed `airflow dags test` on Airflow 3.3.2 with the lab image (dbt `PASS=50`).
