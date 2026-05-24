# DataLab Playground — Claude Instructions

> For full project context, see [AGENTS.md](AGENTS.md). This file contains Claude-specific guidance.

## Post-Change Rule

**After any change to the platform** (versions, services, ports, credentials, architecture, config, or notebooks), you must:
1. Update [README.md](README.md) to reflect the new state.
2. Update [AGENTS.md](AGENTS.md) and [CLAUDE.md](CLAUDE.md) — versions, pitfalls, and any affected section — so future agents have accurate context.

Do not leave these files stale after a change.

## Project Summary

A local Docker Compose–based **AI-enhanced data lakehouse** for experimentation. Services: Spark 4.1.0, Trino 479, Hive Metastore 4.1.0, MinIO, Ollama (LLMs), Qdrant (vectors), Phoenix (AI observability), JupyterLab — all wired together via `docker-compose.yaml`.

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

1. **Two Hive Dockerfiles** — `Dockerfile` (3.1.3, legacy) vs `Dockerfile_4.1` (4.1.0, active). Compose uses a pre-built image; the `build:` block is commented out.
2. **Duplicate spark-defaults.conf** — `spark/conf/` and `jupyter/` must stay in sync.
3. **NVIDIA GPU required** — Ollama won't start without `nvidia-container-toolkit`.
4. **Two Postgres instances** — `phoenix-db` on :5432, `metastore-db` on :5433.
5. **Iceberg JAR** named `iceberg-spark-runtime-4.0_2.13-1.10.1.jar` with Spark 4.1.0 — `4.0` is the Iceberg release's Spark compat version, not a mismatch.

## When Modifying Services

- Spark config changes → update **both** `spark/conf/spark-defaults.conf` and `jupyter/spark-defaults.conf`
- Hive Metastore changes → target `hive-metastore/Dockerfile_4.1` (active) not `Dockerfile`
- After any Dockerfile edit → rebuild: `docker build -t datalab-playground/<service> ./<service>`

## Notebooks Location

- `jupyter/notebooks/data_lab_playground.ipynb` — Platform integration demo
- `jupyter/notebooks/rag_demo.ipynb` — RAG pipeline (Qdrant + Ollama + LangChain)

## Versions Reference

Spark 4.1.0 · Hive 4.1.0 · Hadoop 3.4.1 · Trino 479 · Python 3.12 · Iceberg 1.10.1 · AWS SDK Bundle 2.41.1
