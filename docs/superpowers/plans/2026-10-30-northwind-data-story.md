# One shop, one pipeline: consistent lab data across every notebook

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (inline) or superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Every notebook works on the same fictional company, "Northwind Gear". Its data is the dbt demo shop, so each tool shows its capability as one stage of a single pipeline, not on throwaway data.

**Architecture:** The dbt seeds (customers, orders, products, payments, refunds, product costs, regions) are the system of record. A deterministic module `jupyter/notebooks/lab_data.py` derives everything else from them, so the derived data is the same on every run and stays consistent with the seeds: web clickstream events, delivery coordinates, the live order-event stream, and product documents (manuals, reviews, support tickets). Each notebook loads what it needs from there.

**Tech Stack:** Python 3.12 (stdlib + pandas), Spark 4.1 / Iceberg 1.12, dbt-trino, Trino 483, Qdrant + Ollama (RAG), Airflow 3.3, the Semantic Operator, the MCP servers.

**Spec:** the user's request of 2026-10-30 ("are these consistent and showing the capabilities ... better scenarios for consistency") and the review answer in that conversation: the stage table, the format-v3 constraint, and the three steps below.

## Design

| Stage | Notebook | Shop data | Capability |
|---|---|---|---|
| 0 Tour | `data_lab_playground` | layer row counts, service checks | platform map |
| 1 Ingest | `iceberg/04` | order events streamed into `iceberg_catalog.shop_bronze.order_events` | streaming merge-append, Parquet tuning |
| 2 Clickstream | `iceberg/02` | `lab_data.clickstream()` as VARIANT | VARIANT (format v3) |
| 3 Privacy | `iceberg/03` | a customer's right to be forgotten on a v3 copy of the orders | deletion vectors, row lineage |
| 4 Layout | `iceberg/01` | `lab_data.deliveries()` lat/lon | Hilbert vs Z-order clustering |
| 5 Model | `dbt/01–03` | seeds → staging → marts (unchanged lineage) | dbt on Trino + Iceberg |
| 6 Govern / ask | `semantic/01` | marts | semantic layer over MCP |
| 7 Explain | `rag_demo` | `lab_data.product_docs()` in Qdrant; refund rate per product (semantic) + tickets (RAG) | RAG + metrics |
| 8 Operate | Airflow DAG `northwind_daily` | ingest → `dbt build` → wait for the model to be published | orchestration |
| 9 Observe | `ops/01_pipeline_agent` | the DAG run, the Spark apps | Airflow + Spark History MCP |

Rules:
- **Layers and schemas:** raw = `dbt_demo_raw` (seeds); bronze = `shop_bronze` (Spark-written, format v2, readable by Trino); v3 feature tables = `shop_v3` (Spark writes; Spark and Trino 483 read); models = `dbt_demo`. Notebooks never `DROP SCHEMA` (pitfall 18).
- **Format v3** (VARIANT, deletion vectors, row lineage) stays on `shop_v3` copies, so the bronze layer stays readable by every engine and dbt. Correction from the first draft: Trino 483 and the HMS REST server *do* read v3 here; the original iceberg/02 and 03 notebooks verified it, and the rewrites keep those Trino cells.
- **Determinism:** `lab_data` uses `random.Random(<fixed seed>)` and reads the seed CSVs only. The same output on every machine and run.
- **Referential integrity:**
  - every derived row references real customers, products and orders
  - every order has exactly one checkout event and one delivery
  - support tickets follow the seed's refund pattern (routers returned most, SSDs damaged)
- **Status values:** the raw `placed/shipped/returned` → model `open/fulfilled/returned` mapping stays. It's the staging lesson in `dbt/01`; notebooks name it where they use it.
- `tests/e2e` keeps its own data (tests stay independent).
- Compose (frozen) runs stages 0–5 and 7. Stages 6, 8 and 9 stop with a clear message there.

## Steps

### Step 1: `lab_data.py`, the shared shop data
- [x] Unit tests in `tests/unit/test_lab_data.py`:
  - deterministic (two calls equal)
  - the seeds load
  - `clickstream()`: every event references a known customer and product; exactly one `checkout` per order, with matching `order_id` and timestamp ≤ `order_ts`; payload keys differ by event kind
  - `deliveries()`: one row per order, coordinates inside the customer's country box
  - `order_events()`: every order once, in `order_ts` order, with the raw status
  - `product_docs()`: every product has a manual and reviews; ticket counts per product follow the refunds seed (router and SSD lead)
- [x] Implement `lab_data.py` (stdlib + pandas only; seeds via `lab_utils.DBT_PROJECT`).
- [x] Docs: AGENTS/CLAUDE/README notebooks sections (the module and the layer rules).

### Step 2: Iceberg notebooks on the shop
- [x] `iceberg/04`: stream `lab_data.order_events()` into `shop_bronze.order_events` (rate source mapped onto the events), merge-append, Parquet tuning on that table; Trino reads it.
- [x] `iceberg/02`: `lab_data.clickstream()` into `shop_v3.web_events` as VARIANT; query the funnel (view → cart → checkout) per product.
- [x] `iceberg/03`: copy the orders to `shop_v3.orders`; customer N asks to be forgotten → delete, deletion vectors, row lineage, time-travel audit.
- [x] `iceberg/01`: `lab_data.deliveries()` → Hilbert vs Z-order for "orders near Berlin".
- [x] `data_lab_playground`: the pipeline map and row counts per layer, replacing the random `ecommerce_sales`.
- [ ] Each notebook runs headless on Compose and on the workstation (runbook). **Partly verified; the rest needs the user's machine.**
  - **Ran headless here:** `iceberg/01–04`, `data_lab_playground`, `dbt/01` and `rag_demo` §1 (the numbers), executed with nbclient against a lab-like stack. The stack was Spark Connect 4.1.3, HMS 4.2.1 REST, Trino 483 and SeaweedFS (the chart's images, in Docker), on 2026-10-31. Every Spark/Trino cell passed after two fixes the run found: `iceberg/02` hit the shredded-variant reader bug on a filtered count, and the playground's AI-summary cell needed to survive a missing Ollama.
  - **Not run here** (no GPU/Ollama, no Kubernetes):
    - the LLM sections of the playground, `rag_demo` §2–3 and `ops/01` (the Qdrant indexing and metadata filters were checked with deterministic fake embeddings; the ops MCP helpers were called against real Airflow and Spark History MCP servers)
    - `semantic/01`
    - the Compose classic-Spark mode
  - **Still to run:** the full pass is `hack/workstation/e2e.sh` (runbook step 4) and the Compose notebooks (runbook step 2).

### Step 3: AI and operations on the same shop
- [x] `rag_demo`: index `lab_data.product_docs()`; "Why are router refunds so high?" combines the semantic layer's refund rate per product (chart) or Trino SQL (Compose) with retrieved tickets.
- [x] Airflow DAG `northwind_daily` (shipped with the chart's DAG source and the workstation profile): Spark Connect ingest → `dbt build` (Trino) → wait for `Published` (semantic, chart only). Plus quality checks and `shop_gold.product_kpis`. `airflow dags test` on Airflow 3.3.2 (lab image) passed end to end: ingest replay 10 / 60 events, dbt `PASS=50`, checks 0, gold 10 rows, `published` skipped (no semantic layer in that stack). The wait itself is unit-tested against the CRD's condition shape; it runs for real only on the workstation (`e2e.sh` step 4).
- [x] `ops/01_pipeline_agent`: ask the Airflow and Spark History MCP servers about the last DAG run and its Spark apps.
- [x] Docs and the runbook.

## Outcome and deviations

- **DAG as planned, on a lab Airflow image:**
  - The stock `apache/airflow` image has neither pyspark nor dbt. `airflow/Dockerfile` adds pyspark-client 4.1.3 to Airflow's environment, and dbt 1.12.5 / dbt-trino 1.10.6 in `/opt/dbt`.
  - Chart 0.3.0 adds `airflow.labPipelines.enabled` (off by default; the default render is unchanged). It sets `SPARK_REMOTE`, `TRINO_HOST`/`TRINO_PORT` and `SEMANTIC_MODEL_RESOURCE`. With `semantic.enabled`, it adds a ServiceAccount + Role that may only get/list/watch the release's SemanticModels, which is what `wait_for_published` reads.
  - The workstation profile uses the lab image with `labPipelines` on.
- **Added beyond the plan:** quality checks and the gold table `shop_gold.product_kpis` (refund rate with the semantic model's definition, per product and run date). `data_lab_playground` lists it in *Rows per layer*.
- **Shipping the DAG:**
  - The DAG lives in `jupyter/notebooks/ops/dags/`. The workstation profile mounts the notebooks claim with `subPath: ops/dags`, so the DAG finds `lab_data.py` and the dbt project two folders up.
  - dbt writes its target and logs to a temp dir, because the volume may be read-only (verified with a read-only mount).
  - The DAG is unpaused at creation. `hack/workstation/e2e.sh` triggers it after the dbt notebooks and requires `success`.
- **Ops agent:** `lab_utils.ask_ops` prefixes each tool with its server (`airflow__…`, `spark_history__…`) and routes the call. The tool listing and calls were verified against real MCP servers (34 Airflow tools, 17 Spark History tools). The LLM loop is unit-tested with fake sessions.
- **RAG question:** by amount, the refund rate is highest for tablet and dock (a few expensive returns). By count, tickets lead for router and SSD. So `rag_demo` asks about the two most-ticketed products and shows both columns: the numbers say *what*, the tickets say *why*. The numbers path (Trino over the marts) was verified and matches the seeds exactly.
