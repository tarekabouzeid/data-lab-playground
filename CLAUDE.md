# DataLab Playground — Claude Instructions

> For full project context, see [AGENTS.md](AGENTS.md). This file contains Claude-specific guidance.

## Post-Change Rule

**After any change to the platform** (versions, services, ports, credentials, architecture, config, or notebooks), you must:
1. Update [README.md](README.md) to reflect the new state.
2. Update [AGENTS.md](AGENTS.md) and [CLAUDE.md](CLAUDE.md) — versions, pitfalls, and any affected section — so future agents have accurate context.
3. For any version change, update [docs/VERSIONS.md](docs/VERSIONS.md) (authoritative version matrix) and re-run `tests/e2e/run-e2e.sh`.
4. If the Helm chart ships the changed component, mirror the change in `charts/data-lab`. Then bump the chart version, add a `charts/data-lab/CHANGELOG.md` entry and run `hack/gen-chart-schema.py` + `helm-docs -c charts -s file`.

Do not leave these files stale after a change.

## Release Notes Policy

[RELEASE_NOTES.md](RELEASE_NOTES.md) is the versioned changelog for this project. It is **not** updated automatically after every change.

**Rules for agents:**
- **Never write to `RELEASE_NOTES.md` without explicit user confirmation.**
- Before touching the file, ask the user: *"Is this feature/change complete and ready to be documented in RELEASE_NOTES.md for the next release?"*
- Only proceed if the user says yes.
- When adding a new release entry:
  - Add it at the **top**, below the template comment block, following the existing format exactly.
  - Use the version number and tag the user supplies (e.g. `v3.0.0`).
  - Use `main` as the branch name if merging to main, otherwise use the actual branch name.
  - Set today's date in `YYYY-MM-DD` format.
  - Fill every section; do not leave placeholder text.

**Versioning scheme (for reference when suggesting a version bump):**
- **Major** — breaking changes: service version jumps, architecture changes, removed APIs
- **Minor** — backward-compatible additions: new services, new notebooks, new catalog connectors
- **Patch** — bug fixes, config tweaks, dependency security bumps

## Project Summary

A local Docker Compose–based **AI-enhanced data lakehouse** for experimentation. Services: Spark 4.1.3, Iceberg 1.12.0, Trino 483, Hive Metastore 4.2.1 (Thrift + built-in Iceberg REST catalog), SeaweedFS (S3), Ollama (LLMs), Qdrant (vectors), Phoenix (AI observability), JupyterLab — all wired together via `docker-compose.yaml`. The same lakehouse also ships as one Helm chart (`charts/data-lab`): defaults for many parallel, ephemeral installs (Sandpiper), plus a workstation profile for minikube + GPU (Spark Connect, Spark History Server, Airflow 3, optional MCP servers, Kubeflow Semantic Operator, GenAI, Gateway routes). **Docker Compose is frozen**: version alignment only, new features are chart-only.

## Key Things to Know

- **Start everything**: `./start-platform.sh` (handles image builds + SeaweedFS `warehouse` bucket + Ollama model pull)
- **Force rebuild**: `./start-platform.sh --rebuild`
- **E2E test (no GPU)**: `tests/e2e/run-e2e.sh --build` (19 checks: Spark/Trino × `hive`/`iceberg`/`lakehouse` catalogs + plain-Parquet storage checks)
- **Workstation lab (minikube)**: `hack/workstation/setup-minikube.sh`, `build-images.sh`, `deploy.sh`, `e2e.sh` (charts/data-lab + `ci/workstation-values.yaml`, release/namespace `datalab`, URLs `*.datalab.localhost:8080`); checklist `docs/WORKSTATION_TESTING.md`
- **Jupyter** at http://localhost:8888, password `123456`
- **All Compose credentials are intentionally hardcoded** (local dev only — SeaweedFS S3 `seaweedadmin`/`seaweedadmin123`, Postgres, Jupyter). The Helm chart has none: `existingSecret` or generated.
- **Helm chart**: `charts/data-lab` (README there). Tests: `python3 -m pytest tests/chart tests/unit -q` (render + unit) and `hack/check-compose-parity.py`. Lint with `helm lint --strict charts/data-lab --kube-version 1.34.0`; policy check with `helm template r charts/data-lab -n ns | hack/check-chart-render.py r-data-lab ns`; kind e2e with `hack/chart-e2e-kind.sh minimal|default|parallel|semantic` (local cluster + images: `hack/kind-up.sh`, `--calico` on WSL2 whose kernel lacks the NFQUEUE that kindnet's NetworkPolicy needs).

## Architecture

```
Jupyter (8888) → Spark Master (7077/8081) → SeaweedFS S3 (8333) s3a://warehouse/
Jupyter         → Trino (8080) ─ hive/lakehouse → HMS Thrift (9083) ┐
                                 └ iceberg → HMS Iceberg REST (9084)  ┴→ Postgres (5433) + SeaweedFS
Spark iceberg_catalog → HMS Iceberg REST (9084)  (type=rest, NOT type=hive)
Jupyter         → Ollama (11434) · Qdrant (6333) · Phoenix (6006/4317)
```

## Critical Pitfalls

1. **HMS 4.2.1: Iceberg only via the HMS Iceberg REST catalog** (`http://hive-metastore:9084/iceberg`). HMS ≥ 4.0.1 removed Thrift `get_table`, and Iceberg 1.12.0's `HiveCatalog` (`type=hive`, Hive 2.3.10 client) still calls it → `Invalid method name: 'get_table'`. Spark: `type=rest` + `io-impl=org.apache.iceberg.hadoop.HadoopFileIO`. Trino: `iceberg` catalog = `iceberg.catalog.type=rest`; `hive`/`lakehouse` stay on Thrift 9083. Image `apache/hive:standalone-metastore-4.2.1`; compose uses the pre-built image (build block commented out). An in-place upgrade from 4.0.0 auto-migrates the schema (verified). The REST server embeds Iceberg 1.9.1 (fine for format v2).
2. **Duplicate spark-defaults.conf** — `spark/conf/` and `jupyter/` must stay in sync.
3. **NVIDIA GPU required** — Ollama won't start without `nvidia-container-toolkit`.
4. **Two Postgres instances** — `phoenix-db` on :5432, `metastore-db` on :5433.
5. **Iceberg JAR** named `iceberg-spark-runtime-4.1_2.13-1.12.0.jar`. The artifact ID matches the Spark 4.1 line.
6. **HMS config = `hive-metastore/*.xml.template`**. The entrypoint regenerates `metastore-site.xml`/`core-site.xml` from them on every start; a copied `hive-site.xml` is overwritten. `hive-site.xml` and `entrypoint.sh` there are legacy/unused. Managed warehouse `s3a://warehouse/managed/`, external `s3a://warehouse/` (Hive 4 rejects Iceberg/external tables under the managed root; set both `metastore.*` and `hive.metastore.*` spellings). S3A: bundled `hadoop-aws-3.4.1` + downloaded AWS SDK `bundle-2.24.6` (must match).
7. **HMS path validation** — `hive.metastore.path.validation=false` is set in `metastore-site.xml.template`. Without it, `CREATE TABLE ... external_location 's3a://...'` fails if the S3 path doesn't exist yet. Always use `s3a://` (not `s3://`) in external locations.
8. **Spark capped at 4.1.x**: Iceberg 1.12.0 has no `iceberg-spark-runtime-4.2_2.13`. Keep `spark/Dockerfile`, `jupyter/Dockerfile` `SPARK_VERSION` (also drives `pyspark==`), and the Iceberg jar's Spark line in lockstep. `hadoop-aws` must equal Spark's bundled Hadoop (3.4.2).
9. **Storage is SeaweedFS 4.48** (replaced MinIO, whose image left Docker Hub). Endpoint `http://seaweedfs:8333`, keys in `seaweedfs/s3.json`. The bucket is created via `weed shell` (no `mc`). Keep `-volume.max=64`, because auto-sizing from free disk left no writable volumes and S3A PUTs returned 500. Filer UI is on host 8889.
10. **Trino 483**: the new Web UI is the default at `/ui`. Breaking changes in 482/483 (Alluxio removal, char/varchar coercion, `$files` bounds type, `s3.auth-type`) don't affect current catalogs. Full research: `docs/UPGRADE_PLAN.md`.
11. **Notebooks and scripts never hardcode a service host**: endpoints are `lab_utils.py` constants (env from the chart, Compose host as default); `tests/unit/test_notebook_endpoints.py` enforces it. `hack/check-compose-parity.py` keeps the chart's Spark conf and `charts/data-lab/files/e2e_lakehouse.py` aligned with Compose.
12. **Spark Connect on Kubernetes**: with `SPARK_REMOTE` set, never call `.master(...)` and don't use `SparkContext`/RDDs; static confs live server-side in the chart. Catalog URIs can be overridden at runtime, so read them from env. Kubeflow SDK: `connect(base_url=…)` only; never install `kubeflow[spark]` (hardcoded Spark 4.0.4, pins `pyspark-connect==4.2.0`). Workstation cluster-wide pieces (Spark Operator, Envoy Gateway + GatewayClass, SemanticModel CRD, GPU plugin, notebooks PV) come from `hack/workstation/setup-minikube.sh`, never from the chart. `helm --wait` ignores CRs (SemanticModel, SparkApplication). Details: `AGENTS.md` pitfalls 14–17.

13. **Helm chart (`charts/data-lab`)**: `values.yaml` is a public API (Sandpiper), so only add keys, and give each one a `# --` doc comment. `values.schema.json` is generated (`hack/gen-chart-schema.py`) and closed. Names come only from `datalab.componentName`, and nothing is cluster-scoped. Spark Connect tokens go in `sc://host:15002/;authorization=Bearer <token>`, because `;token=` forces TLS that Spark 4.1 Connect lacks. Chart Postgres is 17.11 (Compose 13). Custom image tags = `appVersion`. Spark Connect must be started with `exec spark-submit` (not `start-connect-server.sh`) so SIGTERM closes the event log for the Spark History Server. The Airflow MCP logs in as its own `viewer`/`op` user, never as admin. Gateway/DRA/semantic need their APIs (`global.apiChecks=false` only for `helm lint`). Semantic Operator: header auth only behind the release NetworkPolicy (C13), one semantic release per namespace (C14), a pre-delete hook removes models before the manager goes, Ossie `ANSI_SQL`, chat model `qwen3:4b`. `tests/chart/golden/default-resources.txt` pins the default render to 0.1.0. Details are in AGENTS.md pitfall 19 and `docs/HELM_GAP_REPORT.md`.

## When Modifying Services

- Spark config changes → update **both** `spark/conf/spark-defaults.conf` and `jupyter/spark-defaults.conf`
- Hive Metastore changes → `hive-metastore/Dockerfile` and `hive-metastore/*.xml.template` (not `hive-site.xml`); the chart's copy is `charts/data-lab/templates/hms-configmap.yaml`
- Image/version/Spark-config changes → also update `charts/data-lab` (see Post-Change Rule) and run `hack/check-compose-parity.py`
- After any Dockerfile edit → rebuild: `docker build -t datalab-playground/<service> ./<service>`

## Notebooks Location

- `jupyter/notebooks/data_lab_playground.ipynb` — Platform integration demo
- `jupyter/notebooks/rag_demo.ipynb` — RAG pipeline (Qdrant + Ollama + LangChain)
- `jupyter/notebooks/iceberg/` (00–04) — Iceberg 1.12 features; `jupyter/notebooks/dbt/` (01–03) + `dbt/lakehouse_demo` project; `jupyter/notebooks/semantic/01_chat_with_your_data.ipynb` (Semantic Operator vs raw text-to-SQL, chart only); shared `lab_utils.py` (all endpoints) and `lab_data.py` (Northwind Gear: the one shop all notebooks use, derived deterministically from the dbt seeds; layers `dbt_demo_raw` → `shop_bronze` (v2) / `shop_v3` (v3, Spark only) → `dbt_demo`). Rules: no module named `dbt_*`, never `DROP SCHEMA` in notebooks, no `.master()` (works on Compose and Spark Connect). The Jupyter image needs `git` for `dbt debug`. See AGENTS.md pitfall 18.

## Versions Reference

Full matrix: [docs/VERSIONS.md](docs/VERSIONS.md).

Spark 4.1.3 (capped: no Iceberg Spark 4.2 runtime) · Iceberg 1.12.0 · Helm chart 0.2.0 (appVersion 3.0.0; adds Airflow 3.3.2, Postgres 17.11, Spark History MCP 1.0.1, astro-airflow-mcp 0.9.1, Kubeflow Semantic Operator @4fd7057 (built from source) — VERSIONS.md §8; workstation tooling §9) · Hive Metastore 4.2.1 · Trino 483 · SeaweedFS 4.48 · Python 3.12 · Java: Spark 21 / Jupyter driver 17 / HMS 21 / Trino 25 · Hadoop 3.4.2 (Spark) / 3.4.1 (HMS) · AWS SDK bundle 2.41.1 (Spark) / 2.24.6 (HMS)
