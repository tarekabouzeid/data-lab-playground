---
description: "Use when editing Spark configuration, spark-defaults.conf, or any Spark settings. Covers the duplicate config file pitfall and key version constraints."
applyTo: "**/spark-defaults.conf"
---

# Spark Configuration Rules

## CRITICAL: Two Identical Config Files — Keep In Sync

The following two files must always be **identical**:
- `spark/conf/spark-defaults.conf` — used by Spark master/worker containers
- `jupyter/spark-defaults.conf` — used by the JupyterLab PySpark driver

**When modifying Spark config, always update both files.**

## Key Versions

- Spark: **4.1.0**
- Hadoop: **3.4.1**
- Iceberg runtime JAR: `iceberg-spark-runtime-4.0_2.13-1.10.1.jar` — the `4.0` in the artifact name is the Iceberg release's Spark compat version, not a version mismatch.
- AWS SDK bundle: `2.41.1`

## S3 / MinIO Settings (Do Not Change for Local Dev)

```
spark.hadoop.fs.s3a.endpoint=http://minio:9000
spark.hadoop.fs.s3a.access.key=minioadmin
spark.hadoop.fs.s3a.secret.key=minioadmin123
spark.hadoop.fs.s3a.path.style.access=true
spark.hadoop.fs.s3a.connection.ssl.enabled=false
```

## Iceberg Catalog Config

Iceberg is registered as `spark_catalog`. Table paths use `s3a://warehouse/`. Do not change the catalog name — notebooks depend on it.
