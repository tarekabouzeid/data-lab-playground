# Release Notes

This file is the canonical changelog for **DataLab Playground**.  
One entry per release, newest at the top.  
Each release maps to a git tag (`vX.Y.Z`) on `main`.

### Versioning scheme
- **Major** (`X`) — breaking changes: service version jumps, architecture changes, removed APIs
- **Minor** (`Y`) — backward-compatible additions: new services, new notebooks, new catalog connectors
- **Patch** (`Z`) — bug fixes, config tweaks, dependency security bumps

---

<!--
## [vX.Y.Z] — YYYY-MM-DD

**Tag:** `vX.Y.Z`  
**Branch merged:** `branch-name`

### Summary
One-paragraph description.

### Component Version Changes
| Component | Before | After |
|---|---|---|

### Breaking Changes
-

### What's New
-

### Bug Fixes / Config Changes
-

### Known Limitations
-
-->

---

## [v2.0.0] — 2026-05-24

**Tag:** `v2.0.0`  
**Branch merged:** `spark-hms-trino-upgrade`

### Summary

Major version upgrade across the entire data lakehouse stack — Spark, Hive Metastore, Trino, Iceberg, and the AWS SDK — plus a full overhaul of the Hive Metastore service and a Jupyter environment cleanup. The RAG notebook was also migrated to be compatible with LangChain 1.x.

---

## Component Version Changes

| Component | `main` (before) | This branch (after) |
|---|---|---|
| Apache Spark | 4.0.1 | **4.1.0** |
| Trino | 476 | **481** |
| Hive Metastore | Custom build — Hive 3.1.3 + Hadoop 3.3.4 + Java 8 | **apache/hive:4.0.0** (official image, held at 4.0.0) |
| Hadoop (Spark/Trino) | 3.3.4 | **3.4.2** |
| Iceberg runtime | `iceberg-spark-runtime-4.0_2.13-1.10.0` | **`iceberg-spark-runtime-4.1_2.13-1.11.0`** |
| Iceberg AWS bundle | 1.10.0 | **1.11.0** |
| AWS SDK Bundle (Spark) | `com.amazonaws:aws-java-sdk-bundle:1.12.367` (SDK v1) | **`software.amazon.awssdk:bundle:2.41.1`** (SDK v2) |
| PostgreSQL JDBC (HMS) | 42.5.1 | **42.7.5** |
| Phoenix DB (postgres) | `postgres` (unpinned) | **`postgres:17`** |
| Jupyter base image | `jupyter/base-notebook:latest` | **`quay.io/jupyter/base-notebook:python-3.12`** (pinned) |
| langchain | unpinned | **1.2.0** |
| langchain-core | unpinned | **1.2.6** |
| langchain-community | unpinned | **0.4.1** |
| langchain-ollama | unpinned | **1.0.1** |
| langchain-qdrant | unpinned | **1.1.0** |
| langchain-text-splitters | unpinned | **1.1.0** |
| langchain-classic | — | **1.0.1** (new, provides legacy chain builders) |
| LangGraph | — | **1.0.5** (new) |
| langsmith | unpinned | **0.6.0** |

---

## Hive Metastore — Full Overhaul

The HMS image was rebuilt from scratch, replacing a hand-assembled OpenJDK 8 + Hive 3.1.3 image with the official `apache/hive:4.0.0`.

**Why 4.0.0 and not a later release?**  
HIVE-26537 removed the legacy `get_table` Thrift method in HMS 4.0.1 and 4.1.0. Iceberg 1.11's shaded Hive 2.3 client calls `get_table` directly; any HMS ≥ 4.0.1 returns `TApplicationException: Invalid method name: 'get_table'`. HMS 4.0.0 is the safe ceiling until Iceberg resolves this upstream (tracked in Iceberg PR #12721).

### Key changes
- **Base image** → `apache/hive:4.0.0` (Debian Bullseye; the `standalone-metastore-4.0.0` tag does not exist)
- **S3A JARs** — the old Dockerfile downloaded `hadoop-aws` from Maven. The new image **symlinks** the already-bundled `/opt/hadoop/share/hadoop/tools/lib/hadoop-aws-3.3.6.jar` and `aws-java-sdk-bundle-1.12.367.jar` into `/opt/hive/lib/`. Downloading `hadoop-aws ≥ 3.4.x` would fail at runtime with `ClassNotFoundException: org.apache.hadoop.fs.BulkDelete` because the image bundles Hadoop 3.3.6 which lacks that class.
- **Custom entrypoint removed** — the official image's built-in entrypoint handles `schematool` schema init.
- **`hive.metastore.path.validation=false`** added to `hive-site.xml` — prevents HMS from rejecting external table `CREATE` statements when the S3 path does not yet exist (e.g. before Spark has written any data).
- **PostgreSQL JDBC** updated from 42.5.1 → 42.7.5.
- **`docker-compose.yaml`** — the HMS `build:` block is commented out; the compose file uses the pre-built image directly.

---

## Spark — 4.0.1 → 4.1.0

- Updated base image `apache/spark:4.0.1` → `apache/spark:4.1.0`.
- **AWS SDK v1 → v2**: replaced `com.amazonaws:aws-java-sdk-bundle:1.12.367` with `software.amazon.awssdk:bundle:2.41.1`.
- **Iceberg artifact ID** corrected from `4.0_2.13` to `4.1_2.13` (the artifact ID now correctly tracks the Spark major.minor version as of Iceberg 1.11).
- **Hadoop AWS** bumped from 3.3.4 → 3.4.2 to match the Spark 4.1 distribution.
- **`spark-defaults.conf`** (both `spark/conf/` and `jupyter/`) — added explicit Python executable paths:
  ```
  spark.pyspark.python=/opt/conda/bin/python
  spark.pyspark.driver.python=/opt/conda/bin/python
  ```
  This ensures Spark workers and the driver use the same Python environment, avoiding version mismatch errors.

---

## Trino — 476 → 481

- Updated base image `trinodb/trino:476` → `trinodb/trino:481`.
- **New catalog: `lakehouse.properties`** — adds a dedicated Iceberg lakehouse catalog backed by the Hive Metastore, with native S3/MinIO support (`fs.native-s3.enabled=true`) and Snappy-compressed Parquet as the default file format.
- **`hive.properties`** — added `hive.non-managed-table-creates-enabled=true` and `hive.non-managed-table-writes-enabled=true` to allow creating and writing external (non-managed) tables.

---

## Jupyter — Environment Cleanup & Pinning

- **Base image pinned**: `jupyter/base-notebook:latest` → `quay.io/jupyter/base-notebook:python-3.12` (sourced from Quay to avoid Docker Hub rate limits; Python version is now explicit).
- **Removed the separate `genai` conda environment** — all packages are now installed directly into the base conda environment, simplifying the kernel setup and eliminating the `JUPYTER_DEFAULT_KERNEL=genai` workaround.
- **All LangChain packages are now fully pinned** — see version table above.
- **Added `langchain-classic==1.0.1`** — required because `langchain ≥ 1.0` removed the `langchain.chains` module. The classic package provides `create_retrieval_chain`, `create_history_aware_retriever`, and `create_stuff_documents_chain`.
- **Added LangGraph** (`langgraph==1.0.5` + checkpointing and prebuilt packages).
- **Added `openinference-instrumentation-langchain==0.1.66`** (pinned) for Phoenix tracing.

---

## Docker Compose

| Change | Detail |
|---|---|
| `metastore-db` | Added host port mapping `5433:5432` and a `pg_isready` healthcheck |
| `phoenix-db` | Pinned to `postgres:17`, added healthcheck, exposed port `5432:5432` (was internal only) |
| `phoenix` | `depends_on` now waits for `phoenix-db` `service_healthy` (was just `service_started`) |
| HMS service | Build block commented out — compose uses the pre-built `apache/hive:4.0.0` image |

---

## RAG Notebook — LangChain 1.x Migration (`rag_demo.ipynb`)

The notebook was updated throughout to be compatible with `langchain==1.2.0`, which removed the `langchain.chains`, `langchain.schema`, `langchain.text_splitter`, and `langchain.memory` modules.

| Old import | New import |
|---|---|
| `from langchain.schema import Document` | `from langchain_core.documents import Document` |
| `from langchain.text_splitter import RecursiveCharacterTextSplitter` | `from langchain_text_splitters import RecursiveCharacterTextSplitter` |
| `from langchain.prompts import PromptTemplate` | `from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder` |
| `from langchain.chains import RetrievalQA` | `from langchain_classic.chains.retrieval import create_retrieval_chain` + `from langchain_classic.chains.combine_documents.stuff import create_stuff_documents_chain` |
| `from langchain.chains import ConversationalRetrievalChain` | `from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever` + `create_retrieval_chain` |
| `from langchain.memory import ConversationBufferMemory` | Removed — `chat_history` is now a plain `list[HumanMessage \| AIMessage]` managed manually |
| `from langchain_community.vectorstores import Qdrant` | Removed (unused; `QdrantVectorStore` from `langchain_qdrant` was already the active integration) |
| `OllamaLLM` | `ChatOllama` (required by the LCEL chain helpers) |

Chain invocation keys also changed: `{"query": ...}` → `{"input": ...}`, and result keys: `result["result"]` → `result["answer"]`, `result["source_documents"]` → `result["context"]`.

---

## New Files

| File | Purpose |
|---|---|
| `AGENTS.md` | Agent-facing project context, pitfall catalogue, version reference |
| `CLAUDE.md` | Claude-specific agent instructions mirroring AGENTS.md |
| `.github/instructions/docker-compose.instructions.md` | Copilot instruction file for docker-compose edits |
| `.github/instructions/dockerfiles.instructions.md` | Copilot instruction file for Dockerfile edits |
| `.github/instructions/spark-config.instructions.md` | Copilot instruction file for Spark config edits |
| `jupyter/Dockerfile-org` | Backup of the original Dockerfile before the upgrade |
| `jupyter/notebooks/trino_query_example_updated.py` | Standalone Trino query example script |
| `trino/etc/catalog/lakehouse.properties` | New Iceberg lakehouse catalog for Trino |

---

## Known Limitations / Notes

- **NVIDIA GPU required** for the `ollama` service (`runtime: nvidia`). The platform will not start without `nvidia-container-toolkit`.
- **All credentials are hardcoded** (MinIO, Postgres, Jupyter password `123456`) — intentional for local dev, never promote to production.
- **Two Postgres instances**: `phoenix-db` on host port `5432`, `metastore-db` on host port `5433`.

---

## [v1.0.0] — 2026-05-01

**Tag:** `v1.0.0`  
**Branch:** `main` (initial release)

### Summary

Initial working platform combining Apache Spark, Trino, Hive Metastore, MinIO, Ollama, Qdrant, and Phoenix under a single `docker-compose.yaml`. Hand-assembled Hive 3.1.3 image, unpinned Jupyter base image, and unpinned LangChain packages.

### Component Versions (at release)

| Component | Version |
|---|---|
| Apache Spark | 4.0.1 |
| Hive Metastore | 3.1.3 (custom build, Hadoop 3.3.4, Java 8) |
| Hadoop | 3.3.4 |
| Trino | 476 |
| Iceberg runtime | `iceberg-spark-runtime-4.0_2.13-1.10.0` |
| AWS SDK Bundle | `com.amazonaws:aws-java-sdk-bundle:1.12.367` (SDK v1) |
| Jupyter base | `jupyter/base-notebook:latest` (unpinned) |
| LangChain | unpinned |
