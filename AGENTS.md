# DataLab Playground — Agent Instructions

A local, Docker Compose–based **AI-enhanced data lakehouse** for experimentation. Combines Apache Spark, Trino, Hive Metastore, SeaweedFS (S3-compatible), and a full GenAI stack (Ollama, LangChain, Phoenix, Qdrant) — all accessible from JupyterLab.

See [README.md](README.md) for a full overview and service access URLs.

---

## Post-Change Rule

**After any change to the platform** (versions, services, ports, credentials, architecture, config, or notebooks), you must:
1. Update [README.md](README.md) to reflect the new state (versions table, service URLs, architecture diagram, or quick-start steps as appropriate).
2. Update [AGENTS.md](AGENTS.md) — specifically the versions table, pitfalls, and any affected section — so future agents have accurate context.
3. For any version change, update [docs/VERSIONS.md](docs/VERSIONS.md) (the authoritative version matrix) and re-run `tests/e2e/run-e2e.sh`.

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

# End-to-end lakehouse test (no GPU needed): SeaweedFS + HMS + Trino + Spark + Iceberg
tests/e2e/run-e2e.sh --build
```

---

## Kubernetes (Helm) Commands

```bash
helm/scripts/setup-minikube.sh [--gpu-mode device-plugin|dra|none]   # cluster + Spark Operator + Envoy Gateway
helm/scripts/build-images.sh && helm/scripts/deploy.sh               # images into minikube, then the chart
helm/scripts/lint.sh [--server-dry-run]                               # static checks + parity (no cluster needed)
helm/scripts/e2e-k8s.sh [--mode application|connect|both]            # the 19-check lakehouse e2e on Kubernetes
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
| Phoenix · Ollama · Qdrant | `arizephoenix/phoenix:version-20.20.0` · `ollama/ollama:0.40.2` · `qdrant/qdrant:v1.19.2` (pinned; were `latest`) |
| Kubernetes (`helm/`) | Kubernetes 1.37 · minikube 1.39.0 · Helm 4.3 · Kubeflow Spark Operator 2.5.2 · Envoy Gateway v1.9.2 · NVIDIA DRA driver 0.5.0 (optional) |

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

7. **Credentials everywhere are plaintext**: Jupyter password (`123456`), SeaweedFS S3, Hive Postgres, Phoenix Postgres — all hardcoded in Dockerfiles and config files. Intentional for local dev only.

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
14. **Helm chart parity (`helm/`)**: the chart must use the *same* images as `docker-compose.yaml` and the same Spark config. `helm/scripts/check-parity.sh` (run by `helm/scripts/lint.sh`) fails on drift in images, `spark/conf/spark-defaults.conf` vs `spark.sparkConf` in `values.yaml`, and `helm/datalab/files/e2e_lakehouse.py` vs `tests/e2e/e2e_lakehouse.py`. Change one side → change the other. Service names on Kubernetes equal the compose service names (the baked-in Trino/HMS/Spark configs rely on that).
15. **Spark on Kubernetes = Spark Connect, not a master**: notebooks get `SPARK_REMOTE=sc://datalab-spark-server:15002`. With `SPARK_REMOTE` set, pyspark 4.1 raises `CANNOT_CONFIGURE_SPARK_CONNECT_MASTER` if the code also calls `.master(...)`, so notebooks must not (compose gets `spark.master` from `spark-defaults.conf`). `SparkContext`/RDD APIs do not exist under Connect. Static confs (`spark.sql.extensions`) cannot be set by a Connect client; they live in the chart's `spark.sparkConf` (server side).
16. **Kubeflow SDK: `connect(base_url=...)` only, never `pip install kubeflow[spark]`.** SDK 0.5.0 hardcodes Spark/image 4.0.4 in create mode and `submit_job()`, and the extra pins `pyspark-connect==4.2.0` (conflicts with `pyspark==4.1.3`). Re-check on every SDK/operator upgrade with the commands in `docs/K8S_HELM_PLAN.md` §1 (blocker row in `docs/VERSIONS.md` §6).
17. **Kubernetes gotchas**: local images need `imagePullPolicy: IfNotPresent` (`:latest` defaults to `Always`); pods use `enableServiceLinks: false` (a Service named `phoenix` injects `PHOENIX_PORT=tcp://…`); SeaweedFS needs a **headless** Service named `seaweedfs`; the `ollama` image has no `curl` (use `ollama list`); the Spark Operator must watch the release namespace (`spark.jobNamespaces`) and the namespace must exist first.

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
  notebooks/                  # Example notebooks
trino/
  Dockerfile                  # Trino 483
  etc/catalog/hive.properties       # Hive connector → HMS thrift
  etc/catalog/iceberg.properties    # Iceberg connector → HMS Iceberg REST catalog (:9084)
  etc/catalog/lakehouse.properties  # Lakehouse connector (Hive + Iceberg tables) → HMS thrift
tests/e2e/
  run-e2e.sh                  # Starts core services, runs the e2e test via spark-submit
  e2e_lakehouse.py            # 19 checks: Spark/Trino × hive/iceberg/lakehouse catalogs
helm/
  README.md                   # Kubernetes quick start, GPU modes, access
  datalab/                    # Helm chart (templates per service, values.yaml, values-minikube.yaml, files/e2e_lakehouse.py)
  scripts/                    # setup-minikube, build-images, deploy, lint, check-parity, e2e-k8s, hosts, port-forward
docs/
  K8S_HELM_PLAN.md            # Design/decisions for the Kubernetes chart
  VERSIONS.md                 # Authoritative version matrix (components, runtimes, access paths, blockers)
  UPGRADE_PLAN.md             # Compatibility research + version pins rationale (2026-10 upgrade)
```

---

## Notebooks

- [`jupyter/notebooks/data_lab_playground.ipynb`](jupyter/notebooks/data_lab_playground.ipynb) — Full platform demo: Phoenix tracing, Ollama LLM, Spark, SeaweedFS, Trino
- [`jupyter/notebooks/rag_demo.ipynb`](jupyter/notebooks/rag_demo.ipynb) — RAG pipeline: Qdrant + `mxbai-embed-large` embeddings + `gemma3:4b` LLM + LangChain
