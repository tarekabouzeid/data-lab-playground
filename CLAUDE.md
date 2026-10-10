# DataLab Playground — Claude Instructions

> For full project context, see [AGENTS.md](AGENTS.md). This file contains Claude-specific guidance.

## Post-Change Rule

**After any change to the platform** (versions, services, ports, credentials, architecture, config, or notebooks), you must:
1. Update [README.md](README.md) to reflect the new state.
2. Update [AGENTS.md](AGENTS.md) and [CLAUDE.md](CLAUDE.md) — versions, pitfalls, and any affected section — so future agents have accurate context.
3. For any version change, update [docs/VERSIONS.md](docs/VERSIONS.md) (authoritative version matrix) and re-run `tests/e2e/run-e2e.sh`.

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

A local Docker Compose–based **AI-enhanced data lakehouse** for experimentation. Services: Spark 4.1.3, Iceberg 1.12.0, Trino 483, Hive Metastore 4.2.1 (Thrift + built-in Iceberg REST catalog), SeaweedFS (S3), Ollama (LLMs), Qdrant (vectors), Phoenix (AI observability), JupyterLab — all wired together via `docker-compose.yaml`.

## Key Things to Know

- **Start everything**: `./start-platform.sh` (handles image builds + SeaweedFS `warehouse` bucket + Ollama model pull)
- **Force rebuild**: `./start-platform.sh --rebuild`
- **E2E test (no GPU)**: `tests/e2e/run-e2e.sh --build` (19 checks: Spark/Trino × `hive`/`iceberg`/`lakehouse` catalogs + plain-Parquet storage checks)
- **Kubernetes**: `helm/` holds a Helm chart with the same images/versions (`helm/README.md`); `helm/scripts/lint.sh` (no cluster) and `helm/scripts/e2e-k8s.sh` (on minikube) are its tests
- **Jupyter** at http://localhost:8888, password `123456`
- **All credentials are intentionally hardcoded** (local dev only — SeaweedFS S3 `seaweedadmin`/`seaweedadmin123`, Postgres, Jupyter)

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
11. **Helm parity**: images and Spark config in `helm/datalab/values.yaml` must equal `docker-compose.yaml` / `spark/conf/spark-defaults.conf` (`helm/scripts/check-parity.sh` enforces it; also copies `tests/e2e/e2e_lakehouse.py`). Phoenix/Ollama/Qdrant are pinned (`version-20.20.0` / `0.40.2` / `v1.19.2`), no longer `latest`.
12. **Spark Connect on Kubernetes**: with `SPARK_REMOTE` set, never call `.master(...)` and don't use `SparkContext`/RDDs; static confs live server-side in the chart. Kubeflow SDK: `connect(base_url=…)` only; never install `kubeflow[spark]` (hardcoded Spark 4.0.4, pins `pyspark-connect==4.2.0`). Details: `AGENTS.md` pitfalls 14–17, `docs/K8S_HELM_PLAN.md`.

## When Modifying Services

- Spark config changes → update **both** `spark/conf/spark-defaults.conf` and `jupyter/spark-defaults.conf`
- Hive Metastore changes → `hive-metastore/Dockerfile` and `hive-metastore/*.xml.template` (not `hive-site.xml`)
- Image/version/Spark-config changes → also update `helm/datalab/values.yaml` (then run `helm/scripts/lint.sh`)
- After any Dockerfile edit → rebuild: `docker build -t datalab-playground/<service> ./<service>`

## Notebooks Location

- `jupyter/notebooks/data_lab_playground.ipynb` — Platform integration demo
- `jupyter/notebooks/rag_demo.ipynb` — RAG pipeline (Qdrant + Ollama + LangChain)

## Versions Reference

Full matrix: [docs/VERSIONS.md](docs/VERSIONS.md).

Spark 4.1.3 (capped: no Iceberg Spark 4.2 runtime) · Iceberg 1.12.0 · Hive Metastore 4.2.1 · Trino 483 · SeaweedFS 4.48 · Python 3.12 · Java: Spark 21 / Jupyter driver 17 / HMS 21 / Trino 25 · Hadoop 3.4.2 (Spark) / 3.4.1 (HMS) · AWS SDK bundle 2.41.1 (Spark) / 2.24.6 (HMS)
