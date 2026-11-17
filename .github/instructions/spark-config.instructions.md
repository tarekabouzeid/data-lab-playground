---
description: "Use when editing Spark configuration, the Spark ConfigMap, or any Spark settings. Covers where the config lives and key version constraints."
applyTo: "charts/data-lab/templates/spark-configmap.yaml"
---

# Spark Configuration Rules

## Where Spark config lives

Spark settings exist once, in `charts/data-lab/templates/spark-configmap.yaml` (Spark Connect, workers and the
Spark History Server read it). There is no `spark-defaults.conf` in the repo any more. The e2e copy of the settings
is `charts/data-lab/files/e2e_lakehouse.py`.

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
