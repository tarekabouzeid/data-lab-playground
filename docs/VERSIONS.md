# Version Matrix

Single source of truth for every version in the platform. **Last verified: 2026-10-03** on branch
`claude/upgrade-iceberg-spark-hive-trino-yc0p1s` (`tests/e2e/run-e2e.sh`: 19/19).
Background research and rejected options: [UPGRADE_PLAN.md](UPGRADE_PLAN.md).

> When you change any version: update this file first, then the short tables in
> [README.md](../README.md), [AGENTS.md](../AGENTS.md) and [CLAUDE.md](../CLAUDE.md), and re-run the e2e test.

## 1. Core lakehouse components

| Component | Version | Image / artifact | Pinned in | Latest upstream (2026-10-03) | Status |
|---|---|---|---|---|---|
| Apache Spark | **4.1.3** | `apache/spark:4.1.3`; `spark-4.1.3-bin-hadoop3.tgz` + `pyspark==4.1.3` (Jupyter) | `spark/Dockerfile`, `jupyter/Dockerfile` (`SPARK_VERSION`) | 4.2.0 | ⛔ **Capped**: no `iceberg-spark-runtime-4.2_2.13` exists yet |
| Apache Iceberg (Spark) | **1.12.0** | `iceberg-spark-runtime-4.1_2.13-1.12.0.jar`, `iceberg-aws-bundle-1.12.0.jar` | `spark/Dockerfile`, `jupyter/Dockerfile` | 1.12.0 | ✅ Latest |
| Hive Metastore | **4.2.1** | `apache/hive:standalone-metastore-4.2.1` | `hive-metastore/Dockerfile` | 4.2.1 | ✅ Latest (Iceberg via the built-in REST catalog) |
| Trino | **483** | `trinodb/trino:483` | `trino/Dockerfile` | 483 | ✅ Latest |
| SeaweedFS (S3 storage) | **4.48** | `chrislusf/seaweedfs:4.48` | `docker-compose.yaml` | 4.48 | ✅ Latest (replaced MinIO; `minio/minio` is gone from Docker Hub) |
| PostgreSQL (HMS backend) | **13** | `postgres:13` | `docker-compose.yaml` (`metastore-db`) | — | Unchanged |

## 2. Runtimes and bundled libraries per container

Read from the images on 2026-10-03.

| Container | Java | Scala | Python | Hadoop client | Iceberg library | Parquet | Hive client | AWS SDK |
|---|---|---|---|---|---|---|---|---|
| `spark-master` / `spark-worker` (`apache/spark:4.1.3`) | **21** (21.0.11) | 2.13.17 | 3.12 (deadsnakes, at `/opt/conda/bin/python`) | 3.4.2 (+ `hadoop-aws-3.4.2`) | 1.12.0 | 1.16.0 | 2.3.10 (bundled; **not** used for Iceberg) | v2 `bundle-2.41.1` (+ 2.54.17 inside `iceberg-aws-bundle`) |
| `jupyter` (Spark driver) | **17** (`openjdk-17-jdk`) | 2.13.17 | 3.12 | 3.4.2 (+ `hadoop-aws-3.4.2`) | 1.12.0 | 1.16.0 | 2.3.10 | v2 `bundle-2.41.1` |
| `hive-metastore` (`standalone-metastore-4.2.1`) | **21** (21.0.3) | — | — | 3.4.1 (+ `hadoop-aws-3.4.1`) | 1.9.1 (embedded REST server) | 1.15.2 | — | v2 `bundle-2.24.6` (downloaded) + Postgres JDBC 42.7.5 |
| `trino` (`trinodb/trino:483`) | **25** (25.0.3) | — | — | Trino native S3 (no Hadoop S3A) | 1.11.0 | 1.17.1 | Trino's own Thrift client | Trino-bundled |

Notes:
- The Jupyter driver runs Java 17 while the executors run Java 21. This works (Spark 4.1 supports 17 and 21), but if you hit serialization oddities,
  align them by switching `spark/Dockerfile` to `apache/spark:4.1.3-java17` or Jupyter to JDK 21.
- Driver and executor **Python must share a minor version** (3.12). Do not move Jupyter to `python-3.13` without the Spark image too.
- Trino's and HMS's Iceberg libraries (1.11.0 / 1.9.1) are older than Spark's (1.12.0). All tables here are format **v2**, which all three handle.
  Avoid format-v3-only features until Trino and Hive ship newer Iceberg.

## 3. Pinned jars and files (keep consistent)

| Artifact | Version | Used by | Must match |
|---|---|---|---|
| `hadoop-aws` | 3.4.2 | Spark, Jupyter | Spark's bundled `hadoop-client-api` (3.4.2) |
| `software.amazon.awssdk:bundle` | 2.41.1 | Spark, Jupyter | works with `hadoop-aws 3.4.2` against SeaweedFS |
| `hadoop-aws` | 3.4.1 (bundled, symlinked) | HMS | HMS image's Hadoop (3.4.1) |
| `software.amazon.awssdk:bundle` | 2.24.6 | HMS | the SDK `hadoop-aws 3.4.1` is built against |
| `iceberg-spark-runtime-4.1_2.13` | 1.12.0 | Spark, Jupyter | Spark **4.1.x** line |
| `iceberg-aws-bundle` | 1.12.0 | Spark, Jupyter | Iceberg version |
| `org.postgresql:postgresql` | 42.7.5 | HMS | — |
| `spark-defaults.conf` | — | `spark/conf/` and `jupyter/` | the two files must be identical |

## 4. Access matrix (what reaches what, and how)

| Client | Catalog / config | Talks to | Protocol | Plain Parquet tables | Iceberg tables |
|---|---|---|---|---|---|
| Trino | `hive` (Hive connector) | HMS `:9083` | Thrift | ✅ read/write (external + managed) | — |
| Trino | `iceberg` (Iceberg connector, `iceberg.catalog.type=rest`) | HMS `:9084/iceberg` | Iceberg REST | — | ✅ read/write, `$snapshots`, `FOR VERSION AS OF` |
| Trino | `lakehouse` (Lakehouse connector) | HMS `:9083` | Thrift | ✅ | ✅ |
| Spark | `iceberg_catalog` (`type=rest`, `io-impl=HadoopFileIO`) | HMS `:9084/iceberg` | Iceberg REST | — | ✅ read/write, snapshots, time travel |
| Spark | path-based `spark.read/write.parquet("s3a://warehouse/...")` | SeaweedFS `:8333` | S3A | ✅ (also reads Trino-written Hive files and Iceberg data files by path) | — |
| HMS, Spark, Trino | — | SeaweedFS `:8333` | S3 (path-style, `seaweedadmin` / `seaweedadmin123`) | storage | storage |

⚠️ Spark `type=hive` (Thrift HiveCatalog) does **not** work with HMS 4.2.1 (`Invalid method name: 'get_table'`).

Every ✅ above is covered by `tests/e2e/run-e2e.sh`.

## 5. Other services (not part of the lakehouse matrix)

| Service | Image | Pinned? |
|---|---|---|
| Jupyter base | `quay.io/jupyter/base-notebook:python-3.12` | Rolling tag (newest JupyterLab 4.x / JupyterHub singleuser at build time) |
| Phoenix | `arizephoenix/phoenix:latest` | ❌ `latest` (must be > 4.0) |
| Phoenix DB | `postgres:17` | ✅ |
| Ollama | `ollama/ollama:latest` | ❌ `latest` |
| Qdrant | `qdrant/qdrant:latest` | ❌ `latest` |
| LangChain stack | `langchain==1.2.0`, `langchain-core==1.2.6`, `langgraph==1.0.5`, … | ✅ pinned in `jupyter/Dockerfile` |

## 6. Upgrade blockers: what to re-check, and when

| Blocked upgrade | Blocked by | Re-check when |
|---|---|---|
| Spark 4.2.x | No `org.apache.iceberg:iceberg-spark-runtime-4.2_2.13` on Maven Central | A new Iceberg release lists a Spark 4.2 runtime |
| `hadoop-aws` 3.4.3 / 3.5.x (Spark) | Must equal the Hadoop bundled in the Spark release (3.4.2 in 4.1.3) | Spark bumps `hadoop.version` |
| Spark Iceberg `type=hive` | Iceberg's HiveCatalog still uses the Hive 2.3.10 client (calls removed `get_table`) | Iceberg ships a Hive-4 client using `get_table_req` |
| Format v3 features | Trino 483 bundles Iceberg 1.11.0; HMS REST embeds Iceberg 1.9.1 | Trino / Hive bump their Iceberg dependency |

Quick check for new releases:
```bash
curl -s https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-spark-runtime-4.2_2.13/maven-metadata.xml | grep -o '<latest>[^<]*'   # 404 = still blocked
curl -s https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-spark-runtime-4.1_2.13/maven-metadata.xml | grep -o '<latest>[^<]*'
curl -s "https://hub.docker.com/v2/repositories/trinodb/trino/tags?page_size=5&ordering=last_updated" | grep -o '"name":"[^"]*"'
curl -s "https://hub.docker.com/v2/repositories/apache/hive/tags?page_size=8&ordering=last_updated" | grep -o '"name":"[^"]*"'
```

## 7. History

| Date | Spark | Iceberg | HMS | Trino | Storage |
|---|---|---|---|---|---|
| 2026-05-24 (v2.0.0) | 4.1.0 | 1.11.0 | 4.0.0 | 481 | MinIO |
| 2026-10-03 (this branch) | 4.1.3 | 1.12.0 | 4.2.1 | 483 | SeaweedFS 4.48 |
