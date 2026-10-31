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
- [ ] **`semantic/01` with the real `qwen3:4b`** (the sandbox used Qwen3-4B-Instruct-2507 as a stand-in). Pass: the
  semantic layer scores 6/6, and the step-by-step cell runs `list_metrics → list_dimensions → query_metric`.
- [ ] **`ops/01` with the real `qwen3:4b`.** Pass: it names the latest `northwind_daily` run and its task states, and
  Spark application ids from the History Server.
- [ ] **`rag_demo` on GPU.** Pass: `numbers from: semantic layer (certified metrics)`; the router answer names firmware
  2.4.1 and the SSD answer the dead-on-arrival drives.
- [ ] **Airflow UI** (`http://airflow.datalab.localhost:8080`): `northwind_daily` is unpaused. The latest run shows
  `build` logging `Done. PASS=50` and `published` succeeding.
- [ ] **Confirm CI on the PR head is green**, including the kind `parallel` job, which was still running at hand-off.
- [ ] **Release:** pick the version/tag, then update `RELEASE_NOTES.md` (agents need your explicit yes).

## Already verified

- CI: chart-lint and the kind minimal, default and semantic jobs passed on 4b494d3. The semantic job also runs the DAG's
  `wait_for_published` against the real Semantic Operator.
- Every notebook ran headless in the sandbox, in Compose mode and in chart (Spark Connect) mode, on CPU with stand-in
  GGUF models from Docker Hub. Details: `docs/superpowers/plans/2026-10-30-northwind-data-story.md`.
- `northwind_daily` passed `airflow dags test` on Airflow 3.3.2 with the lab image (dbt `PASS=50`).
