# Upgrade Plan — Iceberg / Spark / Hive Metastore / Trino / Jupyter (2026-10)

Plan-first: every target version below was checked against upstream release notes,
build files, and the HMS Thrift IDL **before** any code was changed.

## 1. Compatibility research

| Component | Current | Latest upstream | Target | Why |
|---|---|---|---|---|
| Apache Iceberg | 1.11.0 | **1.12.0** (2026-09-29) | **1.12.0** | Ships `iceberg-spark-runtime-4.1_2.13`; built against Spark 4.1.3, Hadoop 3.4.3, Hive client 2.3.10 (`gradle/libs.versions.toml`). |
| Apache Spark | 4.1.0 | 4.2.0 | **4.1.3** | Iceberg 1.12.0 publishes **no** `iceberg-spark-runtime-4.2_2.13` (Maven 404; the release lists only Spark 3.5/4.0/4.1). 4.1.3 is the newest Spark that Iceberg supports and the exact version Iceberg 1.12 is built against. |
| Hive Metastore | 4.0.0 | 4.2.1 | **4.0.0 (held)** | `hive_metastore.thrift` at `rel/release-4.0.1`, `4.1.0`, `4.2.0` and `4.2.1` no longer defines `Table get_table(...)` (HIVE-26537). Only `4.0.0` has it. Iceberg 1.12's `HiveCatalog` still uses the Hive **2.3.10** client (`hive2 = { strictly = "2.3.10" }`), which calls `get_table`, and there is no Hive-4 client module in 1.12 (`settings.gradle` only has `hive-metastore`). Upgrading HMS would break Spark → Iceberg (`Invalid method name: 'get_table'`). Verified empirically in §4. |
| Trino | 481 | **483** | **483** | 482/483 breaking changes reviewed (below); none apply to our catalog configs. |
| Hadoop (`hadoop-aws`, Spark) | 3.4.2 | 3.5.0 | **3.4.2 (held)** | Must match the `hadoop-client-api/runtime` bundled in Spark 4.1.3 (`hadoop.version=3.4.2` in Spark's `pom.xml`). A mismatch causes `NoSuchMethodError`/`ClassNotFoundException`. |
| AWS SDK v2 bundle (Spark) | 2.41.1 | 2.55.x | **2.41.1 (held)** | Works with `hadoop-aws 3.4.2` against MinIO today. Iceberg 1.12 bumped *its own* aws-bundle to 2.54.17, but the platform uses `HadoopFileIO` over `s3a://`, not `S3FileIO`. Not worth the risk of an S3A/SDK checksum regression against MinIO. |
| Jupyter base image | `quay.io/jupyter/base-notebook:python-3.12` | rolling | **unchanged tag (rolling → latest JupyterLab 4.x / JupyterHub singleuser)** | Python must stay **3.12** to match the Spark worker Python (PySpark requires the same minor version on driver and executors). The `python-3.12` tag is rebuilt upstream, so `--rebuild` picks up the latest JupyterLab/JupyterHub. `pyspark` is pinned to `4.1.3` to match the cluster. |
| `pyspark` (Jupyter) | 4.1.0 | 4.2.0 | **4.1.3** | Must equal the cluster Spark version. |

### Trino 482 / 483 breaking changes vs. our config

| Breaking change | Affects us? |
|---|---|
| Alluxio file system / exchange removed | No (we use `fs.native-s3`). |
| `char` ↔ `varchar` coercion reversed | No (`char` is not used in any notebook DDL). |
| `hive.max-initial-splits` / `hive.max-initial-split-size` removed | No (not set). |
| Iceberg `target_max_file_size` / `parquet_writer_row_group_size` session props removed | No (not used). |
| Iceberg `$files.lower_bounds/upper_bounds` now typed rows | No (notebooks only use `$snapshots`). |
| `s3.iam-role` requires `s3.auth-type=IAM_ROLE`; `s3.use-web-identity-token-credentials-provider` removed | No (static `s3.aws-access-key`/`s3.aws-secret-key`). |
| Web UI redesign now at `/ui`, legacy at `/ui/legacy` | Cosmetic. README updated. |

## 2. Change list

1. `spark/Dockerfile`: `apache/spark:4.1.0` → `4.1.3`; Iceberg jars 1.11.0 → 1.12.0.
2. `jupyter/Dockerfile`: `SPARK_VERSION=4.1.3`, Iceberg jars → 1.12.0, remove stale `HIVE_VERSION=4.1.0` env.
3. `trino/Dockerfile`: `trinodb/trino:481` → `483`.
4. `hive-metastore/`: unchanged (4.0.0). Refresh the comments to say why it is still pinned.
5. New `tests/e2e/`: a reproducible end-to-end test (compose override + runner) covering:
   - Spark → Iceberg (HiveCatalog / HMS): create, insert, schema evolution, snapshots, time travel
   - Trino `iceberg` catalog: read Spark-written Iceberg table, `$snapshots`, `FOR VERSION AS OF`, Trino-side INSERT read back by Spark
   - Spark → plain Parquet on MinIO; Trino `hive` catalog external table over it
   - Trino `hive` managed table CTAS
   - Trino `lakehouse` catalog reading both the Hive and the Iceberg table
   - Cross-catalog JOIN `hive` ⋈ `iceberg`
6. Docs: README, AGENTS.md, CLAUDE.md (versions, pitfalls, e2e instructions).
   `RELEASE_NOTES.md` is **not** touched without user confirmation (project policy).

## 3. Explicitly out of scope / rejected

- **Spark 4.2.0**: blocked until Iceberg publishes `iceberg-spark-runtime-4.2_2.13`.
- **HMS 4.1.x / 4.2.x**: blocked until Iceberg ships a Hive-4 compatible client (would need `get_table_req`).
- **Hadoop 3.4.3 / 3.5.0 `hadoop-aws`**: blocked by Spark 4.1.3's bundled Hadoop 3.4.2.
- **Python 3.13**: the Spark image's Python (3.12) and Jupyter's must stay aligned.

## 4. Verification (run 2026-10-03)

`tests/e2e/run-e2e.sh` (needs Docker; no GPU needed — Ollama/Phoenix/Qdrant/Jupyter are not started).

| Run | Stack | Result |
|---|---|---|
| Upgrade target | Spark 4.1.3 + Iceberg 1.12.0 + HMS 4.0.0 + Trino 483 | **15/15 passed** (twice on a clean metastore DB, so re-runs work) |
| Negative check | same, but HMS `apache/hive:4.2.1` | **Fails** on the first Iceberg DDL: `TApplicationException: Invalid method name: 'get_table'` (Trino 483 itself could still create a schema through HMS 4.2.1) |

Caveats from the sandbox the run happened in:
- The build sandbox's egress proxy blocks `apt` inside `docker build`, and blocks `quay.io` and `jdbc.postgresql.org`.
  So the test images were built from the **same base images and the same jar files**
  (`apache/spark:4.1.3` + `hadoop-aws-3.4.2`, `bundle-2.41.1`, `iceberg-spark-runtime-4.1_2.13-1.12.0`,
  `iceberg-aws-bundle-1.12.0`; `apache/hive:4.0.0` + `postgresql-42.7.5` + the S3A symlinks), downloaded on the host.
  The skipped layer is the unchanged deadsnakes Python 3.12 install, so the driver and executors ran on the base image's Python 3.10.
  The Trino image was built from `trino/Dockerfile` unchanged.
- The Jupyter image was **not** built (quay.io blocked). Its changes are version strings only
  (`SPARK_VERSION=4.1.3`, which also drives `pyspark==4.1.3`, confirmed on PyPI; and the same Iceberg 1.12.0 jars as Spark).
- **`minio/minio:latest` no longer exists on Docker Hub** (404 for `minio/minio` and `minio/mc`).
  The e2e run used `pgsty/minio:RELEASE.2026-08-04T00-00-00Z` through a local compose override.
  Whether to change the compose default is an open decision (see AGENTS.md pitfall #11).


---

# Phase 1b — Storage: MinIO → SeaweedFS (applied 2026-10-03)

Why: `minio/minio` and `minio/mc` were removed from Docker Hub (404), so fresh installs could not pull the storage layer.

| Item | Value |
|---|---|
| Image | `chrislusf/seaweedfs:4.48` (2026-09-28), one container running `weed server -filer -s3` |
| S3 endpoint | `http://seaweedfs:8333` (host `localhost:8333`), path-style, no TLS |
| Credentials | `seaweedadmin` / `seaweedadmin123`, identity in `seaweedfs/s3.json` (anonymous → 403) |
| UIs | Master `:9333`, Filer `:8889` (container 8888; host 8888 is Jupyter) |
| Bucket | `warehouse`, created with `weed shell` → `s3.bucket.create -name warehouse` |

Clients changed (endpoint + keys only): Spark `spark-defaults.conf` (both copies), HMS `hive-site.xml` (hadoop-aws 3.3.6 / SDK v1),
Trino `hive`/`iceberg`/`lakehouse` catalogs (native S3), notebooks, and the e2e test.

Issue found and fixed: with `-volume.max=0` SeaweedFS sizes volume slots from free disk (5 GB free → 4 slots). The filer's metadata used
all of them, and every S3A PUT to the bucket's collection failed with HTTP 500 (`No writable volumes and no free volumes left`).
`-volume.max=64` fixes it (volumes are sparse; no space is reserved).

Verification (HMS 4.0.0 stack, Spark Thrift HiveCatalog):
- `tests/e2e/run-e2e.sh`: **19/19**, twice, including the re-run. The 4 new storage checks prove the data on SeaweedFS is plain Parquet:
  the `PAR1` magic on a Trino-written Hive file; Spark reads the Trino-written Hive files and the Iceberg data files directly by path, with no catalog.
- Ad-hoc: the notebook's Spark write settings (S3A fast upload, 5 MB multipart) wrote and read back 2,000,000 rows (~130 MB) of Parquet.
  The notebook's `fs.s3a.committer.name=magic` turns out to be inert: Spark uses `ParquetOutputCommitter`. This is unchanged.
- Trino access: plain Parquet through `hive` (external table and CTAS) and `lakehouse`; Iceberg through `iceberg` and `lakehouse`.
- Note: SeaweedFS 4.48 also ships its own Iceberg REST catalog (`-s3.iceberg.*`). It's not used here, but it's an alternative to the HMS REST catalog in Phase 2.

---

# Phase 2 — Getting past HMS 4.0.0 (designed + spiked 2026-10-03, not applied yet)

## Idea
Stop sending Spark's Iceberg traffic over the HMS Thrift API, which is where the removed `get_table` hurts.
HMS **4.1+** ships a **built-in Iceberg REST catalog** (`hive-standalone-metastore-rest-catalog`,
`metastore.catalog.servlet.*`). Iceberg tables are still stored in HMS, but clients talk Iceberg REST.

```
Spark  ── Iceberg REST ──► HMS 4.2.1 :9084/iceberg ─┐
Trino iceberg ── Iceberg REST (or Thrift) ──────────┤
Trino hive / lakehouse ── Thrift :9083 ─────────────┴─► Postgres + SeaweedFS
```

| Client | Catalog / connector | Transport |
|---|---|---|
| Spark `iceberg_catalog` | `type=rest`, `uri=http://hive-metastore:9084/iceberg`, `io-impl=org.apache.iceberg.hadoop.HadoopFileIO` | REST |
| Trino `iceberg` | Iceberg connector, `iceberg.catalog.type=rest` (recommended), or keep `hive_metastore` | REST (or Thrift) |
| Trino `hive` | Hive connector (unchanged) | Thrift (Trino's own client supports HMS 4.2.1) |
| Trino `lakehouse` | Lakehouse connector (unchanged) | Thrift |

## Spike results (all on `apache/hive:standalone-metastore-4.2.1`)

| Scenario | Result |
|---|---|
| A: Spark REST + Trino `iceberg` Thrift | **15/15** |
| B: Spark REST + Trino `iceberg` REST | **15/15** (and the re-run on existing data passed) |
| Upgrade: data created on HMS 4.0.0, then the image swapped to 4.2.1 on the same DB volume | DB auto-upgraded `4.0.0 → 4.2.0` by `-initOrUpgradeSchema`; old Hive and Iceberg tables readable via `hive`, `iceberg` Thrift, `iceberg` REST and `lakehouse`; a REST write is visible over Thrift |
| Control: Spark Thrift `type=hive` on 4.2.1 | Fails (`Invalid method name: 'get_table'`), as expected |

## What the HMS image needs (each item found during the spike)
1. Base `apache/hive:standalone-metastore-4.2.1`: 1.38 GB vs 2.77 GB, Java 21, Hadoop 3.4.1. Its default template already enables the REST servlet with `auth=none`.
2. Add the **Postgres JDBC** driver and **AWS SDK v2 `bundle-2.24.6`** (the SDK `hadoop-aws-3.4.1` is built against; the image ships neither).
   Symlink `/opt/hadoop/share/hadoop/tools/lib/hadoop-aws-3.4.1.jar` into `/opt/hive/lib` (tools/lib is not on the classpath).
   This replaces the old Hadoop 3.3.6 / SDK v1 pitfall.
3. The entrypoint **regenerates** `metastore-site.xml` and `core-site.xml` from `*.xml.template` on every start (envsubst).
   Ship our config as those **templates**, because a plain `COPY hive-site.xml` is silently overwritten (this is why the earlier
   `apache/hive:4.2.1` test wrote to `file:/opt/hive/data/warehouse`).
4. Hive 4 refuses external tables (all Iceberg tables) under the **managed** root, so split them:
   `metastore.warehouse.dir=s3a://warehouse/managed/`, `metastore.warehouse.external.dir=s3a://warehouse/`.
   The REST servlet reads the legacy `hive.metastore.warehouse.(external.)dir` keys, so set both spellings.
   Existing `s3a://warehouse/<db>.db/...` paths are not under `managed/`, so they keep working.
5. REST port **9084**, next to Thrift on 9083. The upstream default 9001 collided with the old MinIO console; MinIO has since been replaced by SeaweedFS, but 9084 is kept for clarity.
6. Compose: `hive-metastore.depends_on.metastore-db.condition: service_healthy`. The lighter image starts before Postgres is ready,
   so `schematool` exits (a latent race in the current compose too).

## Client-side changes
- `jupyter/notebooks/data_lab_playground.ipynb` Iceberg cell: `type=hive` + Thrift URI → `type=rest` + REST URI + `io-impl`.
- `trino/etc/catalog/iceberg.properties`: `iceberg.catalog.type=rest`, `iceberg.rest-catalog.uri=http://hive-metastore:9084/iceberg`.
  The catalog name stays the same, so notebook SQL is unchanged.
- `tests/e2e`: default to `E2E_SPARK_ICEBERG=rest` (the switch already exists).
- Docs: rewrite pitfall #1 (HMS ceiling) and #6 (S3A jars); add the REST port; update versions.

## Trade-offs / risks to accept
- The REST server embeds **Iceberg 1.9.1** and writes table metadata on each REST commit. Fine for format v2 (what we use);
  newer v3-only features may lag until Hive bumps Iceberg. With Trino on Thrift instead, Trino commits use Trino's own Iceberg library.
- `auth=none` on the REST endpoint: local-dev only, like every other credential here.
- HMS logs `NoClassDefFoundError: org/apache/hadoop/yarn/util/SystemClock` for compaction leader tasks (upstream image gap).
  Harmless here because no Hive ACID tables are used.
- Spark can no longer use `type=hive` Iceberg catalogs, so any user notebook doing that must switch to REST.
