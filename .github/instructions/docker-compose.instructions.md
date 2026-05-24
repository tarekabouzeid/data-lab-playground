---
description: "Use when editing docker-compose.yaml or adding/removing services from the platform. Covers service dependencies, GPU requirements, port conflicts, and profile-gated services."
applyTo: "**/docker-compose.{yaml,yml}"
---

# Docker Compose Rules

## Service Overview (11 containers)

| Service | Image | Critical Notes |
|---|---|---|
| `minio` | `minio/minio:latest` | S3 storage; bucket `warehouse` created by `start-platform.sh` |
| `metastore-db` | `postgres:13` | Hive Metastore backend; host port **5433** |
| `hive-metastore` | pre-built (not built by compose) | build block is commented out; see [dockerfiles instructions](.github/instructions/dockerfiles.instructions.md) |
| `trino` | `datalab-playground/trino:latest` | Single-node; depends on hive-metastore |
| `spark-master` | `datalab-playground/spark:latest` | Coordinator |
| `spark-worker` | `datalab-playground/spark:latest` | 2g RAM, 2 cores |
| `jupyter` | `datalab-playground/jupyter:latest` | Password: `123456` |
| `phoenix` | `arizephoenix/phoenix:latest` | AI observability |
| `db` | `postgres:17` | Phoenix backend; host port **5432** |
| `ollama` | `ollama/ollama:latest` | Requires `runtime: nvidia` — GPU mandatory |
| `ollama-init` | `ollama/ollama:latest` | Profile: `init` — not started by default |
| `qdrant` | `qdrant/qdrant:latest` | Vector DB |

## Key Constraints

- **GPU required**: `ollama` uses `runtime: nvidia`. Never remove this or move to a non-GPU host.
- **`ollama-init` is profile-gated**: runs only with `--profile init`. `start-platform.sh` pulls models via `docker exec` instead.
- **Two Postgres ports**: `db` (phoenix) on host `:5432`, `metastore-db` (hive) on host `:5433`. Do not swap these.
- **Hive Metastore build is commented out**: The `hive-metastore` service uses a pre-built image. Do not uncomment the build block without also verifying the image name.

## Credentials (Hardcoded — Local Dev Only)

| Service | User | Password |
|---|---|---|
| MinIO | `minioadmin` | `minioadmin123` |
| Hive Postgres | `hive` | `hive123` |
| Phoenix Postgres | `postgres` | `postgres` |
| Jupyter | — | `123456` |

**Never promote these credentials to a non-local environment.**
