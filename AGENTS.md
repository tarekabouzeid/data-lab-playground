# DataLab Playground — Agent Instructions

A local, Docker Compose–based **AI-enhanced data lakehouse** for experimentation. Combines Apache Spark, Trino, Hive Metastore, MinIO (S3-compatible), and a full GenAI stack (Ollama, LangChain, Phoenix, Qdrant) — all accessible from JupyterLab.

See [README.md](README.md) for a full overview and service access URLs.

---

## Post-Change Rule

**After any change to the platform** (versions, services, ports, credentials, architecture, config, or notebooks), you must:
1. Update [README.md](README.md) to reflect the new state (versions table, service URLs, architecture diagram, or quick-start steps as appropriate).
2. Update [AGENTS.md](AGENTS.md) — specifically the versions table, pitfalls, and any affected section — so future agents have accurate context.

Do not leave these files stale after a change.

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
```

---

## Service Ports

| Service | URL | Credentials |
|---|---|---|
| Jupyter | http://localhost:8888 | password: `123456` |
| Spark UI (master) | http://localhost:8081 | — |
| Spark UI (worker) | http://localhost:8082 | — |
| Trino | http://localhost:8080 | — |
| MinIO Console | http://localhost:9001 | `minioadmin` / `minioadmin123` |
| MinIO S3 API | http://localhost:9000 | — |
| Ollama API | http://localhost:11434 | — |
| Qdrant | http://localhost:6333 | — |
| Phoenix | http://localhost:6006 | — |
| Hive Metastore (Thrift) | localhost:9083 | — |

---

## Architecture

```
Jupyter → Spark Master/Worker → MinIO (s3a://warehouse/)
Jupyter → Trino → Hive Metastore → MinIO + Postgres (metastore-db :5433)
Jupyter → Ollama (gemma3:4b LLM, mxbai-embed-large embeddings)
Jupyter → Qdrant (vector DB)
Jupyter → Phoenix (AI observability, via gRPC :4317) → Postgres (phoenix-db :5432)
```

---

## Key Component Versions

| Component | Version |
|---|---|
| Apache Spark | 4.1.0 |
| Hive Metastore | 4.0.0 (pre-built image; see pitfall #1) |
| Hadoop | 3.4.2 |
| Trino | 481 |
| Python | 3.12 |
| Iceberg runtime | 1.11.0 (artifact: `iceberg-spark-runtime-4.1_2.13`) |
| AWS SDK bundle | 2.41.1 |

---

## S3 / MinIO Defaults

All services use the same hardcoded credentials (intentional for local dev — **never promote to production**):

- **Endpoint**: `http://minio:9000` (inside Docker) / `http://localhost:9000` (host)
- **Access key**: `minioadmin`  **Secret**: `minioadmin123`
- **Bucket**: `s3a://warehouse/`
- Path-style access: `true`, SSL: disabled

---

## Pitfalls

1. **Single Hive Metastore Dockerfile**: `hive-metastore/Dockerfile` (Hive **4.0.0**). The `docker-compose.yaml` build block is **commented out** and uses a **pre-built image**. **Do not upgrade past 4.0.0** — HIVE-26537 (merged July 2024, PR #3599) removed the legacy `get_table` Thrift method from HMS **4.0.1 AND 4.1.0** (not just 4.2.0 as commonly documented). Iceberg 1.11's shaded Hive 2.3 client calls `get_table` and receives `TApplicationException: Invalid method name: 'get_table'` from any HMS ≥ 4.0.1. **HMS 4.0.0 is the safe ceiling.** The `standalone-metastore-4.0.0` Docker tag does NOT exist; use `apache/hive:4.0.0` (full image, Debian Bullseye). A permanent fix is tracked in Iceberg PR #12721.

2. **Duplicate `spark-defaults.conf`**: Identical files exist at `spark/conf/spark-defaults.conf` and `jupyter/spark-defaults.conf`. **Keep them in sync** when modifying Spark config.

3. **GPU required for Ollama**: `ollama` uses `runtime: nvidia`. The platform will fail to start without an NVIDIA GPU and `nvidia-container-toolkit`.

4. **`ollama-init` is a profile service**: Only runs with `docker compose --profile init up`. The `start-platform.sh` script handles model pulling via `docker exec`.

5. **Two Postgres instances**: `phoenix-db` on host port `5432`, `metastore-db` on host port `5433`.

6. **Iceberg JAR naming**: `iceberg-spark-runtime-4.1_2.13-1.11.0.jar` — the `4.1_2.13` artifact ID now correctly matches Spark 4.1.0. (Prior to Iceberg 1.11, the `4.0_2.13` artifact was used as a workaround.)

7. **Credentials everywhere are plaintext**: Jupyter password (`123456`), MinIO, Hive Postgres, Phoenix Postgres — all hardcoded in Dockerfiles and config files. Intentional for local dev only.

---

## Repository Layout

```
docker-compose.yaml           # All 11 services defined here
start-platform.sh             # One-command startup + smart rebuild detection
hive-metastore/
  Dockerfile                  # Hive 4.0.0 (apache/hive:4.0.0, Debian Bullseye; held at 4.0.0 for Iceberg compat)
  entrypoint.sh               # Waits for Postgres → initSchema → thrift server
  hive-site.xml               # Postgres JDBC + s3a://warehouse/ config
spark/
  Dockerfile                  # Spark 4.1.0 + Hadoop/Iceberg/AWS JARs
  conf/spark-defaults.conf    # Spark cluster config (keep in sync with jupyter/)
jupyter/
  Dockerfile                  # JupyterLab + PySpark + full GenAI stack
  spark-defaults.conf         # Same as spark/conf/spark-defaults.conf
  notebooks/                  # Example notebooks
trino/
  Dockerfile                  # Trino 481
  etc/catalog/lakehouse.properties  # Hive/Iceberg catalog → HMS thrift
```

---

## Notebooks

- [`jupyter/notebooks/data_lab_playground.ipynb`](jupyter/notebooks/data_lab_playground.ipynb) — Full platform demo: Phoenix tracing, Ollama LLM, Spark, MinIO, Trino
- [`jupyter/notebooks/rag_demo.ipynb`](jupyter/notebooks/rag_demo.ipynb) — RAG pipeline: Qdrant + `mxbai-embed-large` embeddings + `gemma3:4b` LLM + LangChain
