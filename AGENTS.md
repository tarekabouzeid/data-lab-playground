# DataLab Playground — Agent Instructions

A local, Docker Compose–based **AI-enhanced data lakehouse** for experimentation. Combines Apache Spark, Trino, Hive Metastore, SeaweedFS (S3-compatible), and a full GenAI stack (Ollama, LangChain, Phoenix, Qdrant) — all accessible from JupyterLab.

See [README.md](README.md) for a full overview and service access URLs.

---

## Post-Change Rule

**After any change to the platform** (versions, services, ports, credentials, architecture, config, or notebooks), you must:
1. Update [README.md](README.md) to reflect the new state (versions table, service URLs, architecture diagram, or quick-start steps as appropriate).
2. Update [AGENTS.md](AGENTS.md) — specifically the versions table, pitfalls, and any affected section — so future agents have accurate context.
3. For any version change, update [docs/VERSIONS.md](docs/VERSIONS.md) (the authoritative version matrix) and re-run `tests/e2e/run-e2e.sh`.
4. If the change touches a component that the Helm chart also ships (image tag, config, port, credential), mirror it in `charts/data-lab` (values, templates, VERSIONS.md §8), bump the chart version, add a `charts/data-lab/CHANGELOG.md` entry, and regenerate the schema and README (`hack/gen-chart-schema.py`, `helm-docs`).

Do not leave these files stale after a change.

---

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

---

## Quick Commands

```bash
# Start everything (builds images if needed)
./start-platform.sh

# Force rebuild all images
./start-platform.sh --rebuild

# Rebuild a single service image
docker build -t datalab-playground/hive-metastore ./hive-metastore
docker build -t datalab-playground/spark ./spark
docker build -t datalab-playground/trino ./trino
docker build -t datalab-playground/jupyter ./jupyter

# View logs
docker compose logs -f [service-name]

# Restart a service
docker compose restart [service-name]

# Helm chart: regenerate schema + README after editing charts/data-lab/values.yaml, then lint
hack/gen-chart-schema.py && helm-docs -c charts -s file
helm lint --strict charts/data-lab --kube-version 1.34.0 -f charts/data-lab/ci/full-values.yaml
helm template r charts/data-lab -n ns | hack/check-chart-render.py r-data-lab ns   # PSS/labels/namespace policy
python3 -m pytest tests/chart tests/unit -q   # render tests (needs helm) + unit tests (lab_utils, parity)
hack/check-compose-parity.py     # Compose spark-defaults vs the chart's rendered conf, e2e script copy
hack/kind-up.sh [--calico]       # local kind cluster + images (--calico on WSL2: kindnet cannot enforce NetworkPolicy there)
hack/chart-e2e-kind.sh minimal   # minimal|default|parallel|semantic on a kind cluster (CI: .github/workflows/chart-e2e.yaml)

# End-to-end lakehouse test (no GPU needed): SeaweedFS + HMS + Trino + Spark + Iceberg
tests/e2e/run-e2e.sh --build
```

---

## Workstation lab (minikube) Commands

charts/data-lab with `ci/workstation-values.yaml`, release and namespace `datalab`. Test checklist: `docs/WORKSTATION_TESTING.md`.

```bash
hack/workstation/setup-minikube.sh [--gpu-mode device-plugin|dra|none]   # cluster + every cluster-wide prerequisite
hack/workstation/build-images.sh && hack/workstation/deploy.sh          # images into minikube, then the chart
hack/workstation/e2e.sh [--skip helm-test,spark-app,notebooks,dag,gateway]  # helm test, SparkApplication 19/19, notebooks, DAG northwind_daily + ops notebook, URLs
hack/workstation/gateway-forward.sh                                      # Envoy Gateway → localhost:8080 (*.datalab.localhost)
hack/workstation/install-headlamp.sh && hack/workstation/headlamp-token.sh   # Headlamp + Kubeflow plugin, login token
```

## Service Ports

| Service | URL | Credentials |
|---|---|---|
| Jupyter | http://localhost:8888 | password: `123456` |
| Spark UI (master) | http://localhost:8081 | — |
| Spark UI (worker) | http://localhost:8082 | — |
| Trino | http://localhost:8080 | — |
| SeaweedFS S3 API | http://localhost:8333 | `seaweedadmin` / `seaweedadmin123` |
| SeaweedFS Filer UI | http://localhost:8889 | — (container port 8888; host 8888 is Jupyter) |
| SeaweedFS Master UI | http://localhost:9333 | — |
| Ollama API | http://localhost:11434 | — |
| Qdrant | http://localhost:6333 | — |
| Phoenix | http://localhost:6006 | — |
| Hive Metastore (Thrift) | localhost:9083 | — |
| HMS Iceberg REST catalog | http://localhost:9084/iceberg | — (no auth; local dev only) |

---

## Architecture

```
Jupyter → Spark Master/Worker → SeaweedFS S3 (s3a://warehouse/)
Jupyter → Trino ─ hive/lakehouse (Thrift :9083) ─┐
Jupyter → Trino ─ iceberg (Iceberg REST :9084) ──┼→ Hive Metastore 4.2.1 → Postgres (metastore-db :5433)
Jupyter → Spark ─ iceberg_catalog (REST :9084) ──┘   (all data on SeaweedFS)
Jupyter → Ollama (gemma3:4b LLM, mxbai-embed-large embeddings)
Jupyter → Qdrant (vector DB)
Jupyter → Phoenix (AI observability, via gRPC :4317) → Postgres (phoenix-db :5432)
```

---

## Key Component Versions

Authoritative matrix: **[docs/VERSIONS.md](docs/VERSIONS.md)** (runtimes, bundled libraries, access matrix, upgrade blockers). Keep it in sync with this table.

| Component | Version |
|---|---|
| Apache Spark | 4.1.3 (capped by Iceberg; see pitfall #10) |
| Apache Iceberg (Spark runtime) | 1.12.0 (artifact: `iceberg-spark-runtime-4.1_2.13`) |
| Hive Metastore | 4.2.1 (`apache/hive:standalone-metastore-4.2.1`, pre-built image; see pitfall #1) |
| Trino | 483 (bundles Iceberg lib 1.11.0) |
| SeaweedFS | 4.48 (`chrislusf/seaweedfs:4.48`; see pitfall #11) |
| Java | Spark image 21 · Jupyter driver 17 · HMS 21 · Trino 25 |
| Python | 3.12 (Jupyter driver and Spark workers must match) |
| Hadoop / `hadoop-aws` | 3.4.2 (Spark, Jupyter) · 3.4.1 (HMS, bundled) |
| AWS SDK v2 bundle | 2.41.1 (Spark, Jupyter) · 2.24.6 (HMS) |
| Postgres JDBC (HMS) | 42.7.5 |
| Jupyter base image | `quay.io/jupyter/base-notebook:python-3.12` (rolling tag), `pyspark[connect]==4.1.3`, `kubeflow==0.5.0` + `kubeflow-spark-api==2.4.0` (**no** `[spark]` extra) |
| Phoenix · Ollama · Qdrant | Compose: `arizephoenix/phoenix:version-20.20.0` · `ollama/ollama:0.40.2` · `qdrant/qdrant:v1.19.2`; chart: non-root tags in VERSIONS.md §8 |
| Helm chart | `charts/data-lab` 0.2.0 (appVersion 3.0.0): Airflow 3.3.2, Postgres 17.11, Spark History MCP 1.0.1, astro-airflow-mcp 0.9.1, Kubeflow Semantic Operator built from source at `4fd7057` (`hack/build-semantic-operator.sh`; CRD from the same commit) |
| Workstation lab | Kubernetes 1.37 · minikube 1.39.0 · Helm 4.x (4.2.4 tested) · Kubeflow Spark Operator 2.5.2 · Envoy Gateway v1.9.2 · Headlamp 0.45.0 + `headlamp_kubeflow` 0.2.0-alpha (pluginctl 0.1.1) · NVIDIA DRA driver 0.5.0 (optional) |

---

## S3 / SeaweedFS Defaults

All services use the same hardcoded credentials (intentional for local dev — **never promote to production**):

- **Endpoint**: `http://seaweedfs:8333` (inside Docker) / `http://localhost:8333` (host)
- **Access key**: `seaweedadmin`  **Secret**: `seaweedadmin123` (identity defined in `seaweedfs/s3.json`)
- **Bucket**: `s3a://warehouse/`
- Path-style access: `true`, SSL: disabled

---

## Pitfalls

1. **Hive Metastore 4.2.1: Iceberg goes through the HMS Iceberg REST catalog, never Thrift `type=hive`**. HIVE-26537 removed the Thrift `get_table` call from HMS **4.0.1, 4.1.0, 4.2.0 and 4.2.1** (checked in `hive_metastore.thrift` per tag). Iceberg 1.12.0's `HiveCatalog` still uses the Hive **2.3.10** client, which calls `get_table` → `TApplicationException: Invalid method name: 'get_table'`. Instead, HMS ≥ 4.1 ships a built-in Iceberg REST catalog, enabled here on **:9084/iceberg** (`auth=none`).
   - **Spark**: `spark.sql.catalog.<name>.type=rest`, `uri=http://hive-metastore:9084/iceberg`, `io-impl=org.apache.iceberg.hadoop.HadoopFileIO` (reuses the `fs.s3a.*` settings).
   - **Trino**: `iceberg` catalog uses `iceberg.catalog.type=rest`; `hive` and `lakehouse` stay on Thrift `:9083` (Trino's own Thrift client supports HMS 4.2.1). All of them see the same tables.
   - Image `apache/hive:standalone-metastore-4.2.1`. The `docker-compose.yaml` build block is **commented out**, and compose uses the **pre-built image** `datalab-playground/hive-metastore:latest`.
   - The REST server embeds **Iceberg 1.9.1** and writes table metadata on REST commits. Fine for format v2; newer v3-only features may lag.
   - Upgrading an existing 4.0.0 install in place works: on first start `schematool -initOrUpgradeSchema` migrates the DB `4.0.0 → 4.2.0`. Verified, with old Parquet and Iceberg tables readable and writable afterwards.
   - HMS logs `NoClassDefFoundError: org/apache/hadoop/yarn/util/SystemClock` for compaction leader tasks (upstream image gap). Harmless: no Hive ACID tables are used. `HMSCatalogServlet ... NoSuchTableException` errors are normal existence checks.

2. **Duplicate `spark-defaults.conf`**: Identical files exist at `spark/conf/spark-defaults.conf` and `jupyter/spark-defaults.conf`. **Keep them in sync** when modifying Spark config.

3. **GPU required for Ollama**: `ollama` uses `runtime: nvidia`. The platform will fail to start without an NVIDIA GPU and `nvidia-container-toolkit`.

4. **`ollama-init` is a profile service**: Only runs with `docker compose --profile init up`. The `start-platform.sh` script handles model pulling via `docker exec`.

5. **Two Postgres instances**: `phoenix-db` on host port `5432`, `metastore-db` on host port `5433`.

6. **Iceberg JAR naming**: `iceberg-spark-runtime-4.1_2.13-1.12.0.jar`. The `4.1_2.13` artifact ID matches the Spark 4.1.x line. (Prior to Iceberg 1.11, the `4.0_2.13` artifact was used as a workaround.)

7. **Credentials everywhere are plaintext in the Compose stack**: Jupyter password (`123456`), SeaweedFS S3, Hive Postgres, Phoenix Postgres — all hardcoded in Dockerfiles and config files. Intentional for local dev only. The Helm chart has **none**: never copy a Compose credential into `charts/`.

8. **HMS config lives in `*.xml.template` files**. The image entrypoint regenerates `metastore-site.xml` and `core-site.xml` from `hive-metastore/metastore-site.xml.template` and `core-site.xml.template` (envsubst) on **every start**, so a copied `hive-site.xml` or `metastore-site.xml` is silently overwritten. (This is why an early 4.2.1 test wrote to `file:/opt/hive/data/warehouse`.) `hive-metastore/hive-site.xml` and `entrypoint.sh` are **legacy and unused**. Other HMS image rules:
   - Hive 4 refuses external tables (all Iceberg tables) under the **managed** root: managed = `s3a://warehouse/managed/`, external = `s3a://warehouse/`. The REST servlet reads the legacy `hive.metastore.warehouse.(external.)dir` names, so both spellings are set.
   - S3A: the image has `hadoop-aws-3.4.1` (in `tools/lib`, symlinked into `/opt/hive/lib`). The Dockerfile adds AWS SDK v2 **`bundle-2.24.6`**, the version hadoop-aws 3.4.1 is built against, plus Postgres JDBC from Maven Central.

9. **HMS path validation**: `hive.metastore.path.validation=false` is set in `metastore-site.xml.template`. Without it, creating a Hive external table with an `s3a://` location fails when the path doesn't exist yet (e.g., before Spark has written data). Also, always use `s3a://` (not `s3://`) in `external_location` — HMS has `fs.s3a.*` but no plain `s3://` FileSystem implementation.

10. **Spark is capped by the Iceberg runtime**: Iceberg 1.12.0 publishes runtimes only for Spark 3.5, 4.0 and 4.1 (no `iceberg-spark-runtime-4.2_2.13` on Maven Central). Do **not** move to Spark 4.2.x until that artifact exists. Keep `spark/Dockerfile` (`apache/spark:<ver>`), `jupyter/Dockerfile` (`SPARK_VERSION`, used for both the tarball and `pyspark==`) and the Iceberg jar's Spark line in lockstep. `hadoop-aws` must stay at the Hadoop version Spark bundles (3.4.2 for Spark 4.1.3).

11. **Object storage is SeaweedFS** (`chrislusf/seaweedfs:4.48`, replaced MinIO in 2026-10 after `minio/minio` disappeared from Docker Hub). One container runs master + volume + filer + S3 gateway (`weed server -s3`). Things to know:
    - S3 identity/keys live in `seaweedfs/s3.json` (mounted read-only). Anonymous requests get 403.
    - The `warehouse` bucket is created with `weed shell` (`s3.bucket.create -name warehouse`) by `start-platform.sh` and `tests/e2e/run-e2e.sh`. There is no `mc`.
    - **Keep `-volume.max=64`**: with the default `-volume.max=0`, slots are sized from free disk (5 GB free → 4 slots), the filer's metadata used all of them, and every S3A PUT failed with HTTP 500 (`No writable volumes and no free volumes left`). Volumes are sparse, so 64 slots reserve no disk.
    - Filer UI container port 8888 is published on host **8889** (host 8888 is Jupyter). Volume server 8080 is not published (host 8080 is Trino).
    - Compose has a healthcheck on `/healthz`; `hive-metastore` waits for it (`service_healthy`).

12. **Trino 482/483 breaking changes** (none affect current configs): Alluxio FS removed; `char`→`varchar` coercion reversed; `hive.max-initial-split*` removed; Iceberg `$files.lower_bounds/upper_bounds` are now typed rows; `s3.iam-role` now needs `s3.auth-type=IAM_ROLE`; the new Web UI is the default at `/ui` (legacy at `/ui/legacy`, disabled by default).

13. **Docker builds need network access**: the Dockerfiles fetch jars from Maven Central (`curl -fsSL`, so they fail loudly on a 404 or 429), including the HMS Postgres JDBC driver and AWS SDK bundle. The Spark and Jupyter images also install packages via `apt`.
14. **Compose parity and frozen Compose**: Docker Compose is frozen (version alignment only; new features are chart-only). `hack/check-compose-parity.py` (chart-lint CI) fails when a `spark/conf/spark-defaults.conf` key is missing from or differs in the chart's rendered Spark conf (allowed differences are listed with a reason in the script), or when `charts/data-lab/files/e2e_lakehouse.py` differs from `tests/e2e/e2e_lakehouse.py`. **Notebooks never hardcode a service host**: every endpoint is a `lab_utils.py` constant (env set by the chart, Compose host as default); `tests/unit/test_notebook_endpoints.py` enforces it, and `tests/chart/test_jupyter.py` checks the chart sets every env var `lab_utils.py` reads.
15. **Spark on Kubernetes = Spark Connect, not a master**: notebooks get `SPARK_REMOTE=sc://<release>-data-lab-spark-connect:15002/;authorization=Bearer $(SPARK_CONNECT_TOKEN)`. With `SPARK_REMOTE` set, pyspark 4.1 raises `CANNOT_CONFIGURE_SPARK_CONNECT_MASTER` if the code also calls `.master(...)`, so notebooks must not (compose gets `spark.master` from `spark-defaults.conf`). `SparkContext`/RDD APIs do not exist under Connect. Static confs (`spark.sql.extensions`) cannot be set by a Connect client; they live in the chart's Spark ConfigMap. Catalog confs (`spark.sql.catalog.*.uri`) *can* be set at runtime, so scripts read them from env (`ICEBERG_REST_URI`), never a hardcoded host.
16. **Kubeflow SDK: `connect(base_url=...)` only, never `pip install kubeflow[spark]`.** SDK 0.5.0 hardcodes Spark/image 4.0.4 in create mode and `submit_job()`, and the extra pins `pyspark-connect==4.2.0` (conflicts with `pyspark==4.1.3`). The chart's Spark Connect is a Deployment, not a `SparkConnect` CR, so `list_sessions()` finds nothing (the notebook treats that as expected). Re-check on every SDK/operator upgrade (`docs/K8S_HELM_PLAN.md` §1, blocker row in `docs/VERSIONS.md` §6).
17. **Workstation (minikube) gotchas**: the chart stays namespaced, so `setup-minikube.sh` installs the cluster-wide pieces: Spark Operator (`spark.jobNamespaces={datalab}`, its own SA/RBAC off), Envoy Gateway **and the GatewayClass `datalab`**, the SemanticModel CRD, the GPU plugin or DRA driver, and the host notebooks folder as a PV + PVC `datalab-notebooks` (the chart only claims it: `jupyter.persistence.existingClaim`, no hostPath in the pod). Images are built with the chart's names (`ghcr.io/tarekabouzeid/data-lab-playground/<name>:<appVersion>`, `IfNotPresent`). **Headlamp** is a separate Helm release (values in `hack/workstation/headlamp-values.yaml`, pins in `versions.env`) whose plugin manager installs plugins **only from Artifact Hub** at pod start; its Kubeflow plugin covers `SparkApplication`, not `SparkConnect`. Trino behind the Gateway needs `http-server.process-forwarded=true` (the chart sets it when `gateway.enabled`; Compose's `trino/etc/config.properties` has it too), or it answers HTTP 406. `helm --wait` ignores custom resources (SemanticModel, SparkApplication): scripts wait on their status themselves. The SparkApplication e2e is applied only after the release is up (in one install the driver reached Trino before it was ready). The previous `helm/datalab` chart was verified end to end on minikube with a GPU (2026-10-27, `docs/K8S_HELM_PLAN.md` §16); the merged chart's workstation run is in `docs/WORKSTATION_TESTING.md`.
18. **New notebooks (Iceberg 1.12 / dbt)**: dbt runs in-process via `lab_utils.dbt()`; never name a Python module `dbt_*` on `sys.path` (dbt imports it as a plugin). Notebooks never `DROP SCHEMA` (a dropped schema cannot be recreated: leftover `managed/<name>.db` dir in SeaweedFS); `reset_dbt_demo()` drops tables/views only. Verified on this stack (Spark Connect, all notebooks twice; Compose path by spark-submit smoke of the same helper): Hilbert `rewrite_data_files`, VARIANT v3 (shredded-variant DML in 1.12.0 needs the vectorization workaround shown in nb 02), deletion vectors (PUFFIN), row lineage (aggregates over lineage columns need the workaround in nb 03), `remove-dangling-deletes`, merge-append for streaming, per-column dictionary encoding, `rewrite_manifests sort_by` (partitioned tables only). **Not supported here**: geometry/geography (REST server answers 406), Spark SQL column defaults, bloom filters, changelog view with delete files. All 10 notebooks run headless (`jupyter nbconvert --execute`) inside the real Compose Jupyter image (2026-10-27). The Jupyter image needs `git`: `dbt debug` checks for it and reports FAILED without it. `rag_demo.ipynb` asks `input()` when the Qdrant collection already has documents, so a headless run must answer that prompt (it otherwise skips the QA chain).

19. **Helm chart (`charts/data-lab`) rules** (full detail in its README; audit in `docs/HELM_GAP_REPORT.md`):
    - `values.yaml` is a **public API** (Sandpiper installs the chart programmatically). Add keys and never rename or remove them without a major chart bump. Every key needs a `# --` comment. Then run `hack/gen-chart-schema.py` (strict schema, closed objects) and `helm-docs -c charts -s file`. CI fails on drift.
    - Bump `Chart.yaml` `version` and add a `charts/data-lab/CHANGELOG.md` entry for every chart change. `appVersion` = lab release, and the `ghcr.io/.../data-lab-playground/*` image tags in values must equal it (the release workflow checks this).
    - Names are always `include "datalab.componentName" (list $ "<component>")`. Never hardcode a name, namespace or host. Hosts come from `datalab.host`. `fullname` is capped at 40 chars so Spark executor names fit.
    - Only namespaced kinds (`hack/check-chart-render.py` enforces it, plus labels, ClusterIP and PSS restricted). Spark executor RBAC is a Role/RoleBinding. Executors get labels and security context from `executor-pod-template.yaml`.
    - Credentials: `existingSecret` or a generated Secret via `datalab.secretValue` (`lookup` keeps it stable across upgrades). Under `helm template` the values regenerate, which is expected. Phoenix's secret needs a digit and a lowercase letter, and the Airflow fernet key is base64 of 32 chars.
    - **Spark Connect token**: the PySpark client sends `;token=` only over TLS, and Spark 4.1 Connect has no TLS. Use `sc://host:15002/;authorization=Bearer <token>`.
    - HMS in Kubernetes: templates are mounted over `/opt/hive/conf/*.xml.template` with `${HMS_DB_PASSWORD}`/`${S3_*}` placeholders (the entrypoint envsubsts them). The `wait-for-s3` init container lists the bucket with Hadoop S3A, because curl 7.76 SigV4 is rejected by SeaweedFS and anonymous 403 cannot tell "missing" from "exists".
    - SeaweedFS creates the bucket in a `postStart` hook (`weed shell -master=localhost:9333`). Its S3 identity is written from the Secret at start.
    - Airflow task logs live on the `<fullname>-airflow-logs` PVC (default on; the api-server reads them locally because LocalExecutor tasks share the pod), with no S3 remote logging. Postgres in the chart is 17.11 (Compose HMS stays on 13). Airflow is 3.3.2 with SimpleAuthManager. Its password file is written by an init container, and the API server is also the execution API for LocalExecutor tasks (`http://localhost:8080/execution/`).
    - Spark History Server (`sparkHistory`, default on) reads `s3a://<bucket>/spark-events`. An `event-log-dir` init container creates it with Spark's own Hadoop client (`FsShell` + a generated `core-site.xml`), because Spark refuses to start if the dir is missing. Spark Connect is started with `exec spark-submit --class org.apache.spark.sql.connect.service.SparkConnectServer`, **not** `sbin/start-connect-server.sh`. That script keeps the JVM a child of bash, so SIGTERM never reaches Spark and the event log is never closed (S3 shows a file only once closed). History `/tmp` must allow exec (zstd native lib).
    - MCP servers (off by default): Kubeflow Spark History MCP (official image, config from a ConfigMap; servers `history` and `live` = Connect UI :4040; DNS-rebinding protection lists the release's Service names). Airflow MCP = `astro-airflow-mcp` in `airflow-mcp/Dockerfile`, logging in as its own SimpleAuthManager user (`viewer` when `readOnly`, else `op`), added to `AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_USERS` and the auth-file init. Neither MCP authenticates callers. The e2e calls both through `hack/mcp-call.py`.
    - **Gateway** (`gateway.enabled`, off): HTTPRoutes per enabled UI on a namespaced Gateway (`gatewayClassName`) or a shared one (`parentRef`), hostnames `{{route}}.{{release}}.{{domain}}` so releases never collide; the chart never creates a GatewayClass. `gateway.allowFromNamespaces` opens the NetworkPolicy to the proxy pods.
    - **API checks**: gateway, `ollama.gpu.dra` and `semantic` fail rendering without their APIs. `helm lint`/`ct lint` cannot pass `--api-versions`, so CI lints with `global.apiChecks=false` (ct.yaml `helm-lint-extra-args`) and renders with `--api-versions` for kubeconform. New namespaced kinds go into `hack/check-chart-render.py`'s allow-list.
    - **Semantic Operator** (`semantic.enabled`, off): upstream manager + server images, configs ported from `kubeflow/semantic-operator@4fd7057` (`templates/_semantic.tpl`), engine = the release's Trino (no password, static identity). The CRD is vendored in `hack/crds/` and installed separately. **C13**: `header` auth trusts `X-Semantic-User`/`X-Semantic-Role`; it is the default only because the release NetworkPolicy is the boundary (use `jwt` + `jwksURL` on shared clusters). **C14**: the manager has no label selector, so at most one semantic-enabled release per namespace (a `lookup` guard in `validate.yaml` refuses a second one). The manager's finalizer would block uninstall, so a **pre-delete hook** (`semantic-cleanup-hook.yaml`, stdlib Python in the Spark image) deletes the release's SemanticModels first and strips the finalizer after 90 s. Ossie expressions are `ANSI_SQL` (no `TRINO` dialect yet, apache/ossie#607); no `spec.views` (Iceberg views via the HMS REST catalog unverified). The MCP model name is `spec.ossie.name` (`dbt_demo`, endpoints key `SEMANTIC_DEMO_MODEL`). The model is the file `charts/data-lab/files/semantic/dbt-demo.yaml` (copy for the notebook: `jupyter/notebooks/semantic/dbt_demo_model.yaml`, kept identical by `hack/check-compose-parity.py`); it needs all seven dbt tables (`fct_orders`, `fct_payments`, `fct_refunds`, `stg_customers`, `dim_products`, `dim_product_costs`, `dim_regions`). The planner is fan-out safe for **ratios and `COUNT(DISTINCT)`** only: a plain `SUM` over orders grouped by a payment/refund dimension still double-counts, so the model's `ai_context` keeps those dimensions with their own metrics. One manager replica without leader election (`Recreate`), so no Lease outlives uninstall.
    - **Chat notebook**: `qwen3:4b` (Ollama tool calling; `gemma3` has none). `lab_utils` speaks MCP with stdlib HTTP (no `mcp` SDK); unit tests fake the LLM, MCP and Trino.
    - **Render tests**: `tests/chart` renders the chart (`tests/chart/conftest.py`); `golden/default-resources.txt` pins the default resource set to chart 0.1.0, so a Sandpiper upgrade with defaults creates and deletes nothing. Regenerate it only for a deliberate default change (and say so in the CHANGELOG).
    - Local verification without a cluster: render, then run each container in Docker with `--user <uid> --cap-drop ALL --security-opt no-new-privileges` and the rendered command/config. Nested kind/k3s does not work in the cloud sandbox (`oom_score_adj` is forbidden).

---

## Repository Layout

```
docker-compose.yaml           # All 11 services defined here
seaweedfs/
  s3.json                     # SeaweedFS S3 identity (seaweedadmin / seaweedadmin123)
start-platform.sh             # One-command startup + smart rebuild detection
hive-metastore/
  Dockerfile                  # HMS 4.2.1 (apache/hive:standalone-metastore-4.2.1) + Postgres JDBC + AWS SDK bundle 2.24.6
  metastore-site.xml.template # Rendered at start: Postgres, warehouse dirs, Iceberg REST :9084
  core-site.xml.template      # Rendered at start: S3A → SeaweedFS
  entrypoint.sh, hive-site.xml # LEGACY, unused (safe to delete)
spark/
  Dockerfile                  # Spark 4.1.3 + Hadoop/Iceberg 1.12.0/AWS JARs
  conf/spark-defaults.conf    # Spark cluster config (keep in sync with jupyter/)
jupyter/
  Dockerfile                  # JupyterLab + PySpark + full GenAI stack
  spark-defaults.conf         # Same as spark/conf/spark-defaults.conf
  notebooks/                  # Example notebooks (iceberg/, dbt/, semantic/, ops/ incl. the Airflow DAG + lab_utils.py: endpoints + helpers, lab_data.py: the Northwind shop)
trino/
  Dockerfile                  # Trino 483
  etc/catalog/hive.properties       # Hive connector → HMS thrift
  etc/catalog/iceberg.properties    # Iceberg connector → HMS Iceberg REST catalog (:9084)
  etc/catalog/lakehouse.properties  # Lakehouse connector (Hive + Iceberg tables) → HMS thrift
tests/e2e/
  run-e2e.sh                  # Starts core services, runs the e2e test via spark-submit
  e2e_lakehouse.py            # 19 checks: Spark/Trino × hive/iceberg/lakehouse catalogs (endpoints from env, Compose defaults)
tests/chart/                  # render tests for charts/data-lab (pytest, needs helm); golden/ = default resource set
tests/unit/                   # lab_utils (semantic chat helpers, endpoints rule), compose parity
charts/data-lab/              # Helm chart (Kubernetes): values.yaml = public API, values.schema.json generated
  templates/                  # one file per component; _helpers.tpl (names, labels, images, secrets), tests/
  ci/*-values.yaml            # profiles: default, minimal, small, ephemeral, full, semantic, workstation (ct lint + kind e2e)
  files/e2e_lakehouse.py      # copy of tests/e2e/e2e_lakehouse.py for the lakehouse helm test
  README.md(.gotmpl), CHANGELOG.md
hack/
  gen-chart-schema.py         # values.yaml -> values.schema.json (--check in CI)
  check-chart-render.py       # policy checks on helm template output
  kind-up.sh                  # local kind cluster (optionally Calico) + image loading for the kind e2e
  chart-e2e-kind.sh           # kind e2e: minimal (timed) / default / parallel / semantic + helm test + clean uninstall
  check-compose-parity.py     # Compose Spark conf + e2e script copy vs the chart
  crds/                       # vendored SemanticModel CRD (Kubeflow Semantic Operator, cluster-scoped)
  workstation/                # minikube lab: setup-minikube, build-images, deploy, e2e, Headlamp, gateway/port forwards, versions.env
  mcp-call.py, airflow-logs-check.py  # in-cluster helpers used by the e2e
airflow-mcp/Dockerfile        # Airflow MCP image (astro-airflow-mcp) for the chart
ct.yaml                       # chart-testing config
.github/workflows/chart-*.yaml # lint, kind e2e, OCI release (images + chart on v* tags)
docs/
  K8S_HELM_PLAN.md            # Design of the former helm/datalab minikube chart (superseded; its real-node findings still apply)
  WORKSTATION_TESTING.md      # Manual test runbook: Compose, kind, workstation, Sandpiper portability
  superpowers/plans/          # implementation plans
  HELM_GAP_REPORT.md          # Helm requirements audit, conflicts with component limits
  VERSIONS.md                 # Authoritative version matrix (components, runtimes, access paths, blockers, §8 chart images, §9 workstation tooling)
  UPGRADE_PLAN.md             # Compatibility research + version pins rationale (2026-10 upgrade)
```

---

## Notebooks

- [`jupyter/notebooks/data_lab_playground.ipynb`](jupyter/notebooks/data_lab_playground.ipynb) — The tour (stage 0): service checks, Ollama + Phoenix, the Northwind pipeline map, `shop_bronze.orders` (Iceberg) + `hive.default.shop_customers` (plain Parquet) joined in Trino, time travel, rows per layer
- [`jupyter/notebooks/rag_demo.ipynb`](jupyter/notebooks/rag_demo.ipynb) — RAG + metrics (stage 7): refund rate per product (semantic layer → Trino over the marts → seeds, first that works) + `lab_data.product_docs()` in Qdrant (`northwind_docs`, rebuilt each run) + `gemma3:4b`; asks why the two most-ticketed products come back
- [`jupyter/notebooks/ops/`](jupyter/notebooks/ops/) — `dags/northwind_daily.py` (stage 8, Airflow 3: check inputs → MERGE raw orders into `shop_bronze.orders_current` → quality checks → `shop_gold.product_kpis`; plain SQL on Trino over HTTP, so the stock Airflow image runs it; `TRINO_HOST` via `airflow.extraEnv`; unit-tested in `tests/unit/test_northwind_dag.py`) and `01_pipeline_agent.ipynb` (stage 9: `lab_utils.ask_ops` gives `qwen3:4b` the tools of the Airflow and Spark History MCP servers; chart only)
- [`jupyter/notebooks/iceberg/`](jupyter/notebooks/iceberg/) — Iceberg 1.12 techniques on the shop: `00` what's new + live support matrix (scratch namespace `iceberg12`), `01` Hilbert vs Z-order on `shop_bronze.deliveries_*`, `02` VARIANT clickstream `shop_v3.web_events` + funnel, `03` right to be forgotten on `shop_v3.orders` (DVs, row lineage, compaction + `expire_snapshots`), `04` order stream into `shop_bronze.order_events` (merge-append, Parquet dictionary per column)
- [`jupyter/notebooks/dbt/`](jupyter/notebooks/dbt/) — dbt (dbt-core 1.12.5 + dbt-trino 1.10.6 on Trino catalog `iceberg`): `01` getting started, `02` incremental merge + snapshots, `03` tests/contracts/unit tests. Project: `jupyter/notebooks/dbt/lakehouse_demo/` (seeds: customers, orders, products, payments, refunds, product costs, regions; the payments/refunds/costs/regions models depend only on their own seeds, so the `stg_orders+` lineage the notebooks show is unchanged)
- [`jupyter/notebooks/lab_data.py`](jupyter/notebooks/lab_data.py) — **Northwind Gear, the one shop every notebook uses** (design: `docs/superpowers/plans/2026-10-30-northwind-data-story.md`). The dbt seeds are the system of record; `lab_data` derives clickstream events, delivery coordinates, the order-event stream and product documents (manuals, reviews, support tickets that follow the refunds seed) deterministically from them. Layers: raw `dbt_demo_raw`, bronze `shop_bronze` (Spark, format v2, Trino-readable), `shop_v3` (format-v3 feature tables: Spark writes, Spark and Trino 483 read), models `dbt_demo`, gold `shop_gold` (the DAG). New notebooks use this data, not ad-hoc random data.
- [`jupyter/notebooks/semantic/01_chat_with_your_data.ipynb`](jupyter/notebooks/semantic/01_chat_with_your_data.ipynb) — walkthrough of the Semantic Operator: the 7-table dbt shop and its traps, the Ossie model, the MCP tools by hand (compiled SQL, fan-out-safe refund rate vs a naive join), the LLM (`qwen3:4b`) step by step, 6 questions semantic vs raw text-to-SQL scored against ground truth, governance (`marketing` denied e-mail, `dach_sales` row filter). Chart with `semantic.enabled` only (stops on Compose).
- `jupyter/notebooks/lab_utils.py` — shared helper (`get_spark()` works on Compose and Spark Connect, `trino()`, `dbt()`, `reset_dbt_demo()`, semantic-layer helpers, `ops_servers()` / `ops_call()` / `ask_ops()` for the operations MCP servers)
