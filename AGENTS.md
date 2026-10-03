# DataLab Playground — Agent Instructions

A local, Docker Compose–based **AI-enhanced data lakehouse** for experimentation. Combines Apache Spark, Trino, Hive Metastore, SeaweedFS (S3-compatible), and a full GenAI stack (Ollama, LangChain, Phoenix, Qdrant) — all accessible from JupyterLab.

See [README.md](README.md) for a full overview and service access URLs.

---

## Post-Change Rule

**After any change to the platform** (versions, services, ports, credentials, architecture, config, or notebooks), you must:
1. Update [README.md](README.md) to reflect the new state (versions table, service URLs, architecture diagram, or quick-start steps as appropriate).
2. Update [AGENTS.md](AGENTS.md) — specifically the versions table, pitfalls, and any affected section — so future agents have accurate context.

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

---

## Architecture

```
Jupyter → Spark Master/Worker → SeaweedFS S3 (s3a://warehouse/)
Jupyter → Trino → Hive Metastore → SeaweedFS + Postgres (metastore-db :5433)
Jupyter → Ollama (gemma3:4b LLM, mxbai-embed-large embeddings)
Jupyter → Qdrant (vector DB)
Jupyter → Phoenix (AI observability, via gRPC :4317) → Postgres (phoenix-db :5432)
```

---

## Key Component Versions

| Component | Version |
|---|---|
| Apache Spark | 4.1.3 (capped by Iceberg; see pitfall #10) |
| Hive Metastore | 4.0.0 (pre-built image; see pitfall #1) |
| Hadoop (Spark/Trino) | 3.4.2 |
| Hadoop (bundled in HMS image) | 3.3.6 |
| Trino | 483 |
| Python | 3.12 |
| Iceberg runtime | 1.12.0 (artifact: `iceberg-spark-runtime-4.1_2.13`) |
| Jupyter base image | `quay.io/jupyter/base-notebook:python-3.12` (rolling tag), `pyspark==4.1.3` |
| AWS SDK bundle (Spark/Trino) | 2.41.1 |
| AWS SDK bundle (HMS, via symlink) | 1.12.367 (SDK v1, bundled in apache/hive:4.0.0) |

---

## S3 / SeaweedFS Defaults

All services use the same hardcoded credentials (intentional for local dev — **never promote to production**):

- **Endpoint**: `http://seaweedfs:8333` (inside Docker) / `http://localhost:8333` (host)
- **Access key**: `seaweedadmin`  **Secret**: `seaweedadmin123` (identity defined in `seaweedfs/s3.json`)
- **Bucket**: `s3a://warehouse/`
- Path-style access: `true`, SSL: disabled

---

## Pitfalls

1. **Single Hive Metastore Dockerfile**: `hive-metastore/Dockerfile` (Hive **4.0.0**). The `docker-compose.yaml` build block is **commented out** and uses a **pre-built image**. **Do not upgrade past 4.0.0** — HIVE-26537 (merged July 2024, PR #3599) removed the legacy `get_table` Thrift method from HMS **4.0.1 AND 4.1.0** (not just 4.2.0 as commonly documented). Iceberg's `HiveCatalog` (1.11 and 1.12) uses the Hive 2.3.10 client bundled with Spark, which calls `get_table` and receives `TApplicationException: Invalid method name: 'get_table'` from any HMS ≥ 4.0.1. **HMS 4.0.0 is the safe ceiling.** The `standalone-metastore-4.0.0` Docker tag does NOT exist; use `apache/hive:4.0.0` (full image, Debian Bullseye). A permanent fix is tracked in Iceberg PR #12721.
   - **Re-verified 2026-10 for Iceberg 1.12.0 / HMS 4.2.1**: `hive_metastore.thrift` has no `Table get_table(...)` at tags `rel/release-4.0.1`, `4.1.0`, `4.2.0`, `4.2.1`; Iceberg 1.12.0 still pins the Hive client to `2.3.10` (`hive2 = { strictly = "2.3.10" }`) and has no Hive-4 module. Running `tests/e2e/run-e2e.sh` against `apache/hive:4.2.1` fails at the first Iceberg DDL with `Invalid method name: 'get_table'`. (Trino 483 itself can talk to HMS 4.2.1. Only Spark/Iceberg is blocked.)
   - Re-check this before any future HMS bump: the blocker is gone only once Iceberg's `HiveCatalog` uses `get_table_req`.

2. **Duplicate `spark-defaults.conf`**: Identical files exist at `spark/conf/spark-defaults.conf` and `jupyter/spark-defaults.conf`. **Keep them in sync** when modifying Spark config.

3. **GPU required for Ollama**: `ollama` uses `runtime: nvidia`. The platform will fail to start without an NVIDIA GPU and `nvidia-container-toolkit`.

4. **`ollama-init` is a profile service**: Only runs with `docker compose --profile init up`. The `start-platform.sh` script handles model pulling via `docker exec`.

5. **Two Postgres instances**: `phoenix-db` on host port `5432`, `metastore-db` on host port `5433`.

6. **Iceberg JAR naming**: `iceberg-spark-runtime-4.1_2.13-1.12.0.jar`. The `4.1_2.13` artifact ID matches the Spark 4.1.x line. (Prior to Iceberg 1.11, the `4.0_2.13` artifact was used as a workaround.)

7. **Credentials everywhere are plaintext**: Jupyter password (`123456`), SeaweedFS S3, Hive Postgres, Phoenix Postgres — all hardcoded in Dockerfiles and config files. Intentional for local dev only.

8. **HMS S3A JARs: do NOT download hadoop-aws ≥ 3.4.x into the HMS image**. `apache/hive:4.0.0` bundles Hadoop **3.3.6** in `/opt/hadoop/`. `hadoop-aws-3.4.x` requires `org.apache.hadoop.fs.BulkDelete` (added in Hadoop 3.4.0) — absent in 3.3.6 → `ClassNotFoundException` at runtime, causing HMS to close the Thrift socket mid-request. The HMS Dockerfile instead **symlinks** the already-bundled `/opt/hadoop/share/hadoop/tools/lib/hadoop-aws-3.3.6.jar` and `aws-java-sdk-bundle-1.12.367.jar` into `/opt/hive/lib/`.

9. **HMS path validation**: `hive.metastore.path.validation=false` is set in `hive-site.xml`. Without it, creating a Hive external table with an `s3a://` location fails when the path doesn't exist yet (e.g., before Spark has written data). Also, always use `s3a://` (not `s3://`) in `external_location` — HMS has `fs.s3a.*` but no plain `s3://` FileSystem implementation.

10. **Spark is capped by the Iceberg runtime**: Iceberg 1.12.0 publishes runtimes only for Spark 3.5, 4.0 and 4.1 (no `iceberg-spark-runtime-4.2_2.13` on Maven Central). Do **not** move to Spark 4.2.x until that artifact exists. Keep `spark/Dockerfile` (`apache/spark:<ver>`), `jupyter/Dockerfile` (`SPARK_VERSION`, used for both the tarball and `pyspark==`) and the Iceberg jar's Spark line in lockstep. `hadoop-aws` must stay at the Hadoop version Spark bundles (3.4.2 for Spark 4.1.3).

11. **Object storage is SeaweedFS** (`chrislusf/seaweedfs:4.48`, replaced MinIO in 2026-10 after `minio/minio` disappeared from Docker Hub). One container runs master + volume + filer + S3 gateway (`weed server -s3`). Things to know:
    - S3 identity/keys live in `seaweedfs/s3.json` (mounted read-only). Anonymous requests get 403.
    - The `warehouse` bucket is created with `weed shell` (`s3.bucket.create -name warehouse`) by `start-platform.sh` and `tests/e2e/run-e2e.sh`. There is no `mc`.
    - **Keep `-volume.max=64`**: with the default `-volume.max=0`, slots are sized from free disk (5 GB free → 4 slots), the filer's metadata used all of them, and every S3A PUT failed with HTTP 500 (`No writable volumes and no free volumes left`). Volumes are sparse, so 64 slots reserve no disk.
    - Filer UI container port 8888 is published on host **8889** (host 8888 is Jupyter). Volume server 8080 is not published (host 8080 is Trino).
    - Compose has a healthcheck on `/healthz`; `hive-metastore` waits for it (`service_healthy`).

12. **Trino 482/483 breaking changes** (none affect current configs): Alluxio FS removed; `char`→`varchar` coercion reversed; `hive.max-initial-split*` removed; Iceberg `$files.lower_bounds/upper_bounds` are now typed rows; `s3.iam-role` now needs `s3.auth-type=IAM_ROLE`; the new Web UI is the default at `/ui` (legacy at `/ui/legacy`, disabled by default).

13. **Docker builds need network access**: the Dockerfiles fetch jars from Maven Central (`curl -fsSL`, so they fail loudly on a 404 or 429) and packages via `apt`. The HMS image fetches the Postgres JDBC driver from `jdbc.postgresql.org`.

---

## Repository Layout

```
docker-compose.yaml           # All 11 services defined here
seaweedfs/
  s3.json                     # SeaweedFS S3 identity (seaweedadmin / seaweedadmin123)
start-platform.sh             # One-command startup + smart rebuild detection
hive-metastore/
  Dockerfile                  # Hive 4.0.0 (apache/hive:4.0.0, Debian Bullseye; held at 4.0.0 for Iceberg compat)
  entrypoint.sh               # Waits for Postgres → initSchema → thrift server
  hive-site.xml               # Postgres JDBC + s3a://warehouse/ config
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
  etc/catalog/iceberg.properties    # Iceberg connector (hive_metastore catalog type)
  etc/catalog/lakehouse.properties  # Lakehouse connector (Hive + Iceberg tables) → HMS thrift
tests/e2e/
  run-e2e.sh                  # Starts core services, runs the e2e test via spark-submit
  e2e_lakehouse.py            # 19 checks: Spark/Trino × hive/iceberg/lakehouse catalogs
docs/
  UPGRADE_PLAN.md             # Compatibility research + version pins rationale (2026-10 upgrade)
```

---

## Notebooks

- [`jupyter/notebooks/data_lab_playground.ipynb`](jupyter/notebooks/data_lab_playground.ipynb) — Full platform demo: Phoenix tracing, Ollama LLM, Spark, SeaweedFS, Trino
- [`jupyter/notebooks/rag_demo.ipynb`](jupyter/notebooks/rag_demo.ipynb) — RAG pipeline: Qdrant + `mxbai-embed-large` embeddings + `gemma3:4b` LLM + LangChain
