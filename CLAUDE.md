# DataLab Playground — Claude Instructions

> For full project context, see [AGENTS.md](AGENTS.md). This file contains Claude-specific guidance.

## Post-Change Rule

**After any change to the platform** (versions, services, ports, credentials, architecture, config, or notebooks), you must:
1. Update [README.md](README.md) to reflect the new state.
2. Update [AGENTS.md](AGENTS.md) and [CLAUDE.md](CLAUDE.md) — versions, pitfalls, and any affected section — so future agents have accurate context.

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

A local Docker Compose–based **AI-enhanced data lakehouse** for experimentation. Services: Spark 4.1.3, Iceberg 1.12.0, Trino 483, Hive Metastore 4.0.0, SeaweedFS (S3), Ollama (LLMs), Qdrant (vectors), Phoenix (AI observability), JupyterLab — all wired together via `docker-compose.yaml`.

## Key Things to Know

- **Start everything**: `./start-platform.sh` (handles image builds + SeaweedFS `warehouse` bucket + Ollama model pull)
- **Force rebuild**: `./start-platform.sh --rebuild`
- **E2E test (no GPU)**: `tests/e2e/run-e2e.sh --build` (19 checks: Spark/Trino × `hive`/`iceberg`/`lakehouse` catalogs + plain-Parquet storage checks)
- **Jupyter** at http://localhost:8888, password `123456`
- **All credentials are intentionally hardcoded** (local dev only — SeaweedFS S3 `seaweedadmin`/`seaweedadmin123`, Postgres, Jupyter)

## Architecture

```
Jupyter (8888) → Spark Master (7077/8081) → SeaweedFS S3 (8333) s3a://warehouse/
Jupyter         → Trino (8080) → Hive Metastore (9083) → Postgres (5433) + SeaweedFS
Jupyter         → Ollama (11434) · Qdrant (6333) · Phoenix (6006/4317)
```

## Critical Pitfalls

1. **Single Hive Dockerfile** — `hive-metastore/Dockerfile` (**4.0.0**). Compose uses a pre-built image; the `build:` block is commented out. **Do not upgrade past 4.0.0** — HIVE-26537 removed `get_table` from the Thrift IDL in HMS **4.0.1, 4.1.0, 4.2.0 and 4.2.1** (checked in the Thrift IDL per tag). Iceberg's `HiveCatalog` (still pinned to the Hive **2.3.10** client in Iceberg 1.12.0) calls `get_table` and gets `TApplicationException: Invalid method name: 'get_table'`. Re-verified empirically against `apache/hive:4.2.1` in 2026-10. HMS 4.0.0 is the safe ceiling. Use `apache/hive:4.0.0` (full image, Debian Bullseye — no `standalone-metastore-4.0.0` tag exists).
2. **Duplicate spark-defaults.conf** — `spark/conf/` and `jupyter/` must stay in sync.
3. **NVIDIA GPU required** — Ollama won't start without `nvidia-container-toolkit`.
4. **Two Postgres instances** — `phoenix-db` on :5432, `metastore-db` on :5433.
5. **Iceberg JAR** named `iceberg-spark-runtime-4.1_2.13-1.12.0.jar`. The artifact ID matches the Spark 4.1 line.
6. **HMS S3A JARs — do NOT download `hadoop-aws ≥ 3.4.x`** into the HMS image. `apache/hive:4.0.0` bundles Hadoop **3.3.6**; `hadoop-aws-3.4.x` requires `BulkDelete` (absent in 3.3.6) → `ClassNotFoundException` → Thrift socket closed. The Dockerfile **symlinks** `/opt/hadoop/share/hadoop/tools/lib/hadoop-aws-3.3.6.jar` and `aws-java-sdk-bundle-1.12.367.jar` into `/opt/hive/lib/`.
7. **HMS path validation** — `hive.metastore.path.validation=false` is set in `hive-site.xml`. Without it, `CREATE TABLE ... external_location 's3a://...'` fails if the S3 path doesn't exist yet. Always use `s3a://` (not `s3://`) in external locations.
8. **Spark capped at 4.1.x**: Iceberg 1.12.0 has no `iceberg-spark-runtime-4.2_2.13`. Keep `spark/Dockerfile`, `jupyter/Dockerfile` `SPARK_VERSION` (also drives `pyspark==`), and the Iceberg jar's Spark line in lockstep. `hadoop-aws` must equal Spark's bundled Hadoop (3.4.2).
9. **Storage is SeaweedFS 4.48** (replaced MinIO, whose image left Docker Hub). Endpoint `http://seaweedfs:8333`, keys in `seaweedfs/s3.json`. The bucket is created via `weed shell` (no `mc`). Keep `-volume.max=64`, because auto-sizing from free disk left no writable volumes and S3A PUTs returned 500. Filer UI is on host 8889.
10. **Trino 483**: the new Web UI is the default at `/ui`. Breaking changes in 482/483 (Alluxio removal, char/varchar coercion, `$files` bounds type, `s3.auth-type`) don't affect current catalogs. Full research: `docs/UPGRADE_PLAN.md`.

## When Modifying Services

- Spark config changes → update **both** `spark/conf/spark-defaults.conf` and `jupyter/spark-defaults.conf`
- Hive Metastore changes → target `hive-metastore/Dockerfile`
- After any Dockerfile edit → rebuild: `docker build -t datalab-playground/<service> ./<service>`

## Notebooks Location

- `jupyter/notebooks/data_lab_playground.ipynb` — Platform integration demo
- `jupyter/notebooks/rag_demo.ipynb` — RAG pipeline (Qdrant + Ollama + LangChain)

## Versions Reference

Spark 4.1.3 · Hive 4.0.0 · Hadoop 3.4.2 (Spark/Trino) / 3.3.6 (HMS bundled) · Trino 483 · Python 3.12 · Iceberg 1.12.0 · SeaweedFS 4.48 · AWS SDK Bundle 2.41.1 (Spark/Trino) / 1.12.367 (HMS)
