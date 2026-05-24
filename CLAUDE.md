# DataLab Playground — Claude Instructions

> For full project context, see [AGENTS.md](AGENTS.md). This file contains Claude-specific guidance.

## Post-Change Rule

**After any change to the platform** (versions, services, ports, credentials, architecture, config, or notebooks), you must:
1. Update [README.md](README.md) to reflect the new state.
2. Update [AGENTS.md](AGENTS.md) and [CLAUDE.md](CLAUDE.md) — versions, pitfalls, and any affected section — so future agents have accurate context.

Do not leave these files stale after a change.

## Project Summary

A local Docker Compose–based **AI-enhanced data lakehouse** for experimentation. Services: Spark 4.1.0, Trino 481, Hive Metastore 4.0.0, MinIO, Ollama (LLMs), Qdrant (vectors), Phoenix (AI observability), JupyterLab — all wired together via `docker-compose.yaml`.

## Key Things to Know

- **Start everything**: `./start-platform.sh` (handles image builds + MinIO bucket + Ollama model pull)
- **Force rebuild**: `./start-platform.sh --rebuild`
- **Jupyter** at http://localhost:8888, password `123456`
- **All credentials are intentionally hardcoded** (local dev only — MinIO, Postgres, Jupyter)

## Architecture

```
Jupyter (8888) → Spark Master (7077/8081) → MinIO s3a://warehouse/
Jupyter         → Trino (8080) → Hive Metastore (9083) → Postgres (5433) + MinIO
Jupyter         → Ollama (11434) · Qdrant (6333) · Phoenix (6006/4317)
```

## Critical Pitfalls

1. **Single Hive Dockerfile** — `hive-metastore/Dockerfile` (**4.0.0**). Compose uses a pre-built image; the `build:` block is commented out. **Do not upgrade past 4.0.0** — HIVE-26537 removed `get_table` from the Thrift IDL in HMS **4.0.1 AND 4.1.0** (not just 4.2.0). Iceberg 1.11's shaded Hive 2.3 client calls `get_table` and gets `TApplicationException: Invalid method name: 'get_table'`. HMS 4.0.0 is the safe ceiling. Use `apache/hive:4.0.0` (full image, Debian Bullseye — no `standalone-metastore-4.0.0` tag exists).
2. **Duplicate spark-defaults.conf** — `spark/conf/` and `jupyter/` must stay in sync.
3. **NVIDIA GPU required** — Ollama won't start without `nvidia-container-toolkit`.
4. **Two Postgres instances** — `phoenix-db` on :5432, `metastore-db` on :5433.
5. **Iceberg JAR** named `iceberg-spark-runtime-4.1_2.13-1.11.0.jar` — artifact ID now correctly matches Spark 4.1 (resolved as of Iceberg 1.11).
6. **HMS S3A JARs — do NOT download `hadoop-aws ≥ 3.4.x`** into the HMS image. `apache/hive:4.0.0` bundles Hadoop **3.3.6**; `hadoop-aws-3.4.x` requires `BulkDelete` (absent in 3.3.6) → `ClassNotFoundException` → Thrift socket closed. The Dockerfile **symlinks** `/opt/hadoop/share/hadoop/tools/lib/hadoop-aws-3.3.6.jar` and `aws-java-sdk-bundle-1.12.367.jar` into `/opt/hive/lib/`.
7. **HMS path validation** — `hive.metastore.path.validation=false` is set in `hive-site.xml`. Without it, `CREATE TABLE ... external_location 's3a://...'` fails if the S3 path doesn't exist yet. Always use `s3a://` (not `s3://`) in external locations.

## When Modifying Services

- Spark config changes → update **both** `spark/conf/spark-defaults.conf` and `jupyter/spark-defaults.conf`
- Hive Metastore changes → target `hive-metastore/Dockerfile`
- After any Dockerfile edit → rebuild: `docker build -t datalab-playground/<service> ./<service>`

## Notebooks Location

- `jupyter/notebooks/data_lab_playground.ipynb` — Platform integration demo
- `jupyter/notebooks/rag_demo.ipynb` — RAG pipeline (Qdrant + Ollama + LangChain)

## Versions Reference

Spark 4.1.0 · Hive 4.0.0 · Hadoop 3.4.2 (Spark/Trino) / 3.3.6 (HMS bundled) · Trino 481 · Python 3.12 · Iceberg 1.11.0 · AWS SDK Bundle 2.41.1 (Spark/Trino) / 1.12.367 (HMS)
