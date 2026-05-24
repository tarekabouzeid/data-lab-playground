---
description: "Use when editing Dockerfiles for any service in this project — hive-metastore, spark, trino, or jupyter. Covers which files are active, version constraints, and JAR naming."
applyTo: "**/Dockerfile*"
---

# Dockerfile Rules

## Hive Metastore — Two Dockerfiles, Only One Is Active

| File | Hive Version | Status |
|---|---|---|
| `hive-metastore/Dockerfile` | 3.1.3 | **Legacy — not used by docker-compose** |
| `hive-metastore/Dockerfile_4.1` | 4.1.0 | **Active (used as pre-built image)** |

- The `build:` block in `docker-compose.yaml` is **commented out**.
- The compose service uses the pre-built image `datalab-playground/hive-metastore:latest`.
- **Target `Dockerfile_4.1`** for any Hive Metastore changes.
- After editing, rebuild: `docker build -t datalab-playground/hive-metastore -f hive-metastore/Dockerfile_4.1 ./hive-metastore`

## `jupyter/Dockerfile-org` Is Superseded

`jupyter/Dockerfile-org` is an older version using conda envs. The active file is `jupyter/Dockerfile`.

## Shared JAR Versions (Must Be Consistent Across Services)

| JAR | Version |
|---|---|
| `hadoop-aws` | `3.4.1` |
| `aws-java-sdk-bundle` | `2.41.1` |
| `iceberg-spark-runtime-4.0_2.13` | `1.10.1` |
| `iceberg-aws-bundle` | `1.10.1` |
| PostgreSQL JDBC | `42.7.5` (Hive 4.1 Dockerfile) |

If upgrading a JAR, update it in **all** Dockerfiles that include it (spark, jupyter, hive-metastore).

## Rebuild After Any Dockerfile Change

```bash
docker build -t datalab-playground/hive-metastore -f hive-metastore/Dockerfile_4.1 ./hive-metastore
docker build -t datalab-playground/spark ./spark
docker build -t datalab-playground/trino ./trino
docker build -t datalab-playground/jupyter ./jupyter
```
