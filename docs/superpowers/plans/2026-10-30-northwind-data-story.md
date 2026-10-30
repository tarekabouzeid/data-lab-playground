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
- **Layers and schemas:** raw = `dbt_demo_raw` (seeds); bronze = `shop_bronze` (Spark-written, format v2, readable by Trino); v3 feature demos = `shop_v3` (Spark only); models = `dbt_demo`. Notebooks never `DROP SCHEMA` (pitfall 18).
- **Format v3** (VARIANT, deletion vectors, row lineage) stays on `shop_v3` copies: Trino 483 and the HMS REST server can't read v3 yet. Each notebook says why.
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
- [ ] Unit tests in `tests/unit/test_lab_data.py`:
  - deterministic (two calls equal)
  - the seeds load
  - `clickstream()`: every event references a known customer and product; exactly one `checkout` per order, with matching `order_id` and timestamp ≤ `order_ts`; payload keys differ by event kind
  - `deliveries()`: one row per order, coordinates inside the customer's country box
  - `order_events()`: every order once, in `order_ts` order, with the raw status
  - `product_docs()`: every product has a manual and reviews; ticket counts per product follow the refunds seed (router and SSD lead)
- [ ] Implement `lab_data.py` (stdlib + pandas only; seeds via `lab_utils.DBT_PROJECT`).
- [ ] Docs: AGENTS/CLAUDE/README notebooks sections (the module and the layer rules).

### Step 2: Iceberg notebooks on the shop
- [ ] `iceberg/04`: stream `lab_data.order_events()` into `shop_bronze.order_events` (rate source mapped onto the events), merge-append, Parquet tuning on that table; Trino reads it.
- [ ] `iceberg/02`: `lab_data.clickstream()` into `shop_v3.web_events` as VARIANT; query the funnel (view → cart → checkout) per product.
- [ ] `iceberg/03`: copy the orders to `shop_v3.orders`; customer N asks to be forgotten → delete, deletion vectors, row lineage, time-travel audit.
- [ ] `iceberg/01`: `lab_data.deliveries()` → Hilbert vs Z-order for "orders near Berlin".
- [ ] `data_lab_playground`: the pipeline map and row counts per layer, replacing the random `ecommerce_sales`.
- [ ] Each notebook runs headless on Compose and on the workstation (runbook).

### Step 3: AI and operations on the same shop
- [ ] `rag_demo`: index `lab_data.product_docs()`; "Why are router refunds so high?" combines the semantic layer's refund rate per product (chart) or Trino SQL (Compose) with retrieved tickets.
- [ ] Airflow DAG `northwind_daily` (shipped with the chart's DAG source and the workstation profile): Spark Connect ingest → `dbt build` (Trino) → wait for `Published` (semantic, chart only).
- [ ] `ops/01_pipeline_agent`: ask the Airflow and Spark History MCP servers about the last DAG run and its Spark apps.
- [ ] Docs and the runbook.
