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

Authoritative list: [docs/VERSIONS.md](../../docs/VERSIONS.md).

- Spark: **4.1.3** (capped: Iceberg has no Spark 4.2 runtime yet)
- Hadoop / `hadoop-aws`: **3.4.2** (must equal Spark's bundled Hadoop client)
- Iceberg runtime JAR: `iceberg-spark-runtime-4.1_2.13-1.12.0.jar` (+ `iceberg-aws-bundle-1.12.0.jar`)
- AWS SDK v2 bundle: `2.41.1`
- Python 3.12 on driver and executors; Java 21 in the Spark image, 17 in the Jupyter driver

## Iceberg Catalog (Spark)

Use the Hive Metastore's built-in Iceberg REST catalog. Never use `type=hive`: HMS 4.2.1 removed the Thrift `get_table` call it needs.

```
spark.sql.catalog.iceberg_catalog=org.apache.iceberg.spark.SparkCatalog
spark.sql.catalog.iceberg_catalog.type=rest
spark.sql.catalog.iceberg_catalog.uri=http://hive-metastore:9084/iceberg
spark.sql.catalog.iceberg_catalog.io-impl=org.apache.iceberg.hadoop.HadoopFileIO
```

## S3 / SeaweedFS Settings (Do Not Change for Local Dev)

```
spark.hadoop.fs.s3a.endpoint=http://seaweedfs:8333
spark.hadoop.fs.s3a.access.key=seaweedadmin
spark.hadoop.fs.s3a.secret.key=seaweedadmin123
spark.hadoop.fs.s3a.path.style.access=true
spark.hadoop.fs.s3a.connection.ssl.enabled=false
```

## Iceberg Catalog Config

Iceberg is registered as `spark_catalog`. Table paths use `s3a://warehouse/`. Do not change the catalog name — notebooks depend on it.
