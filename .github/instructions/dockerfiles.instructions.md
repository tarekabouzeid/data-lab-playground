---
description: "Use when editing Dockerfiles for any service in this project — hive-metastore, spark, trino, or jupyter. Covers which files are active, version constraints, and JAR naming."
applyTo: "**/Dockerfile*"
---

# Dockerfile Rules

## Hive Metastore Dockerfile

`hive-metastore/Dockerfile` — Hive **4.0.0**, based on `apache/hive:4.0.0` (Debian Bullseye). This is the only active Dockerfile. **Do not upgrade past 4.0.0** — HIVE-26537 removed the legacy `get_table` Thrift API in HMS 4.0.1 AND 4.1.0 (not just 4.2.0), breaking Iceberg's HiveTableOperations with `TApplicationException: Invalid method name: 'get_table'`. HMS 4.0.0 is the safe ceiling.

- The `build:` block in `docker-compose.yaml` is **commented out**.
- The compose service uses the pre-built image `datalab-playground/hive-metastore:latest`.
- After editing, rebuild: `docker build -t datalab-playground/hive-metastore ./hive-metastore`

## `jupyter/Dockerfile-org` Is Superseded

`jupyter/Dockerfile-org` is an older version using conda envs. The active file is `jupyter/Dockerfile`.

## Shared JAR Versions (Must Be Consistent Across Services)

| JAR | Version |
|---|---|
| `hadoop-aws` | `3.4.2` |
| `aws-java-sdk-bundle` | `2.41.1` |
| `iceberg-spark-runtime-4.1_2.13` | `1.11.0` |
| `iceberg-aws-bundle` | `1.11.0` |
| PostgreSQL JDBC | `42.7.5` (HMS Dockerfile) |

If upgrading a JAR, update it in **all** Dockerfiles that include it (spark, jupyter, hive-metastore).

## Rebuild After Any Dockerfile Change

```bash
docker build -t datalab-playground/hive-metastore ./hive-metastore
docker build -t datalab-playground/spark ./spark
docker build -t datalab-playground/trino ./trino
docker build -t datalab-playground/jupyter ./jupyter
```
