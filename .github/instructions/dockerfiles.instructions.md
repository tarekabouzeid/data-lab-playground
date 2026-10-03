---
description: "Use when editing Dockerfiles for any service in this project — hive-metastore, spark, trino, or jupyter. Covers which files are active, version constraints, and JAR naming."
applyTo: "**/Dockerfile*"
---

# Dockerfile Rules

## Hive Metastore Dockerfile

`hive-metastore/Dockerfile` — Hive Metastore **4.2.1**, based on `apache/hive:standalone-metastore-4.2.1` (Java 21, Hadoop 3.4.1).

- Spark reaches Iceberg tables through the HMS **built-in Iceberg REST catalog** on `:9084/iceberg`, because HMS ≥ 4.0.1 removed the Thrift `get_table` call that Iceberg's Hive 2.3.10 client (`type=hive`) needs. Never configure Spark Iceberg catalogs with `type=hive` against this HMS.
- Config lives in `metastore-site.xml.template` / `core-site.xml.template`. The image entrypoint regenerates the real files from these on every start, so a copied `hive-site.xml` is silently overwritten. `hive-site.xml` and `entrypoint.sh` in that folder are legacy and unused.
- The image adds the Postgres JDBC driver and AWS SDK v2 `bundle-2.24.6` (matching the bundled `hadoop-aws-3.4.1`), and symlinks `hadoop-aws` from `tools/lib` into `/opt/hive/lib`. Keep the SDK version matched to the image's `hadoop-aws`.
- The `build:` block in `docker-compose.yaml` is **commented out**.
- The compose service uses the pre-built image `datalab-playground/hive-metastore:latest`.
- After editing, rebuild: `docker build -t datalab-playground/hive-metastore ./hive-metastore`

## `jupyter/Dockerfile-org` Is Superseded

`jupyter/Dockerfile-org` is an older version using conda envs. The active file is `jupyter/Dockerfile`.

## Shared JAR Versions (Must Be Consistent Across Services)

Authoritative matrix: [docs/VERSIONS.md](../../docs/VERSIONS.md).

| JAR | Version |
|---|---|
| `hadoop-aws` (Spark, Jupyter) | `3.4.2` |
| AWS SDK v2 `bundle` (Spark, Jupyter; saved as `aws-java-sdk-bundle-2.41.1.jar`) | `2.41.1` |
| `iceberg-spark-runtime-4.1_2.13` | `1.12.0` |
| `iceberg-aws-bundle` | `1.12.0` |
| PostgreSQL JDBC | `42.7.5` (HMS Dockerfile) |
| AWS SDK v2 bundle (HMS only) | `2.24.6` (matches HMS's `hadoop-aws-3.4.1`; independent of Spark's 2.41.1) |

If upgrading a JAR, update it in **all** Dockerfiles that include it (spark, jupyter, hive-metastore).

## Rebuild After Any Dockerfile Change

```bash
docker build -t datalab-playground/hive-metastore ./hive-metastore
docker build -t datalab-playground/spark ./spark
docker build -t datalab-playground/trino ./trino
docker build -t datalab-playground/jupyter ./jupyter
```
