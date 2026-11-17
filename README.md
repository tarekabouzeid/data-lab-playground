# DataLab Playground

A Kubernetes lakehouse for exploring data analytics and AI tools, shipped as one Helm chart (`charts/data-lab`): Spark, Iceberg, Trino, Hive Metastore, S3 storage, Airflow, and an optional AI stack (any OpenAI-compatible LLM or a local Ollama, Qdrant, Phoenix, MCP servers, a semantic layer). (Compose removed in chart 0.4.0.)

## Architecture Overview

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   Jupyter   │  │   Phoenix   │  │   Ollama    │  │    Trino    │
│  Notebook   │  │AI Observ.   │  │ LLM Server  │  │   Engine    │
└─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
       │                │                │               │
       └────────────────┼────────────────┼───────────────┘
                        │                │
              ┌─────────────┐    ┌─────────────┐
              │    Spark    │    │    Hive     │
              │   Cluster   │    │ Metastore   │
              └─────────────┘    └─────────────┘
                        │                │
                        └────────────────┘
                                │
                      ┌─────────────┐    ┌─────────────┐
                      │  SeaweedFS  │    │   Qdrant    │
                      │  (S3 API)   │    │ Vector DB   │
                      └─────────────┘    └─────────────┘
```


### Data Processing & Storage
- **SeaweedFS**: S3-compatible object storage (`s3a://warehouse/`); data is stored as plain Parquet files, plus Iceberg tables whose data files are also Parquet
- **Apache Spark**: Basic data processing capabilities  
- **Hive Metastore 4.2.1**: table metadata. Thrift on `:9083` (Trino `hive`/`lakehouse`) and a built-in **Iceberg REST catalog** on `:9084/iceberg` (Spark, Trino `iceberg`)
- **Trino**: SQL query interface

### AI & ML Tools
- **LLM**: any OpenAI-compatible endpoint (OpenRouter, a hosted API, ...) or a local Ollama (needs a GPU); or none, and the notebooks skip their AI parts
- **Phoenix**: Basic AI monitoring 
- **Qdrant**: Vector database for AI experiments
- **Jupyter**: Notebook environment with common libraries

### Infrastructure
- **PostgreSQL**: Database backend 
- **Kubernetes**: kind, minikube or any cluster ≥ 1.29; a GPU only for the local Ollama

### Component Versions

Full matrix (runtimes, bundled libraries, access paths, upgrade blockers): **[docs/VERSIONS.md](docs/VERSIONS.md)**.

| Component | Version | Latest? | Notes |
|---|---|---|---|
| Apache Spark | 4.1.3 | ⛔ capped (4.2.0 exists) | Iceberg has no Spark 4.2 runtime yet |
| Apache Iceberg | 1.12.0 | ✅ | `iceberg-spark-runtime-4.1_2.13` + `iceberg-aws-bundle` |
| Hive Metastore | 4.2.1 | ✅ | Thrift `:9083` + built-in Iceberg REST catalog `:9084/iceberg` |
| Trino | 483 | ✅ | `hive` + `lakehouse` → Thrift, `iceberg` → Iceberg REST |
| SeaweedFS | 4.48 | ✅ | S3 on `:8333` (replaced MinIO) |
| Java | Spark 21 · Jupyter driver 17 · HMS 21 · Trino 25 | | from the images |
| Python | 3.12 | | must match between the Jupyter driver and the Spark workers |
| Hadoop / `hadoop-aws` | 3.4.2 (Spark) · 3.4.1 (HMS) | | each matches its runtime's bundled Hadoop |
| AWS SDK v2 bundle | 2.41.1 (Spark) · 2.24.6 (HMS) | | |
| Jupyter base | `quay.io/jupyter/base-notebook:python-3.12` | rolling | `pyspark[connect]==4.1.3`, `kubeflow==0.5.0` (no `[spark]` extra) |
| Phoenix · Ollama · Qdrant | non-root tags pinned by the chart | pinned | see docs/VERSIONS.md §8 |
| Helm chart `charts/data-lab` | 0.4.0 (appVersion 3.0.0) · Kubernetes ≥ 1.29 · Kubeflow Semantic Operator (`4fd7057`, built from source) | | see [docs/VERSIONS.md](docs/VERSIONS.md) §8 |
| Workstation lab (`hack/workstation/`) | K8s 1.37 · minikube 1.39 · Helm 4.x · Spark Operator 2.5.2 · Envoy Gateway v1.9.2 · Headlamp 0.45.0 | | §9 |

See [docs/UPGRADE_PLAN.md](docs/UPGRADE_PLAN.md) for the compatibility research behind these pins.


## 🚀 Quick Start

### Prerequisites
- A Kubernetes cluster ≥ 1.29 (kind or minikube locally) and Helm 3.19+ (or 4.x)
- 8GB+ RAM for the default profile, 12GB+ for the workstation profile; 50GB+ free disk
- For AI: either an OpenAI-compatible endpoint and key (no GPU needed), or an NVIDIA GPU for a local Ollama, or neither

## ☸️ Kubernetes (Helm)

The lakehouse ships as one Helm chart, [`charts/data-lab`](charts/data-lab/README.md). Its defaults suit many
short-lived, isolated installs side by side (one per user or agent; Sandpiper); a profile turns it into a workstation lab.

- Spark runs as a **Spark Connect** server (`sc://...:15002`) with executors as pods, plus a **Spark History Server**.
  **Airflow 3** (LocalExecutor) is included. Optional and off by default: MCP servers for agents (Kubeflow Spark
  History MCP, Astronomer Airflow MCP), the **Kubeflow Semantic Operator** (an Apache Ossie semantic layer on Trino,
  served over MCP), Jupyter, the GenAI services (Ollama, Qdrant, Phoenix) and Gateway API routes for browsers.
- There are no hardcoded credentials: each one comes from an `existingSecret` or is generated once per release.
- Everything the chart renders is namespaced and runs non-root (Pod Security Standard "restricted"). Cluster-wide
  prerequisites (Spark Operator, Envoy Gateway, the SemanticModel CRD, GPU plugins) are installed separately.
- The `<release>-data-lab-endpoints` ConfigMap lists every endpoint and the Secret holding its credential.

```bash
helm install lab1 oci://ghcr.io/tarekabouzeid/charts/data-lab --version 0.4.0 -n lab1 --create-namespace --wait
helm test lab1 -n lab1        # Spark writes an Iceberg table, Trino reads it back
helm uninstall lab1 -n lab1   # leaves nothing behind
```

### Workstation lab (minikube + NVIDIA GPU)

```bash
hack/workstation/setup-minikube.sh      # minikube + Spark Operator + Envoy Gateway + SemanticModel CRD + Headlamp (--gpu-mode none for CPU only)
hack/workstation/build-images.sh        # build the lab images inside minikube
hack/workstation/deploy.sh              # charts/data-lab with ci/workstation-values.yaml (release/namespace datalab)
hack/workstation/e2e.sh                 # helm test, the 19 checks as a SparkApplication, every notebook, the northwind_daily DAG, every URL
hack/workstation/gateway-forward.sh     # http://jupyter.datalab.localhost:8080, trino.…, airflow.…, headlamp.…
```

Jupyter edits the repo's `jupyter/notebooks` folder directly. Headlamp (with the Kubeflow plugin) logs in with a token
from `hack/workstation/headlamp-token.sh`. Test checklist: [docs/WORKSTATION_TESTING.md](docs/WORKSTATION_TESTING.md).

Profiles, values, credentials, security and resource footprint are in the [chart README](charts/data-lab/README.md).
CI (`.github/workflows/chart-*.yaml`) runs render tests, lint and kind e2e scenarios (minimal, default, parallel,
semantic). The audit behind the chart is in [docs/HELM_GAP_REPORT.md](docs/HELM_GAP_REPORT.md).

## 🎯 Getting Started

1. **Deploy**: `hack/workstation/deploy.sh` (see above) and `hack/workstation/gateway-forward.sh`
2. **Open Jupyter**: http://jupyter.datalab.localhost:8080 (token: see Workstation lab)
3. **Try the demos**: open `data_lab_playground.ipynb`, the tour of Northwind Gear (the one shop every notebook uses) and its pipeline
4. **Experiment with RAG**: `rag_demo.ipynb`
5. **Explore UIs**: Trino, Airflow, Spark History, Phoenix, Qdrant, Headlamp on `*.datalab.localhost:8080`

## 🌐 Service Access Points

Everything is routed through the Gateway on `*.datalab.localhost:8080` (workstation profile). Ports that are not HTTP
(HMS Thrift 9083, Spark Connect 15002, OTLP 4317) come from `hack/workstation/port-forward.sh`. Every endpoint and the
Secret that holds its credential is listed in the `<release>-data-lab-endpoints` ConfigMap.

| Service | Host | Notes |
|---------|------|-------|
| JupyterLab | `jupyter.datalab.localhost` | token from the `<release>-data-lab-jupyter` Secret |
| Trino UI | `trino.datalab.localhost` | no authentication |
| Airflow | `airflow.datalab.localhost` | DAG `northwind_daily` |
| Spark History | `spark-history.datalab.localhost` | |
| Phoenix | `phoenix.datalab.localhost` | AI traces |
| Qdrant | `qdrant.datalab.localhost` | `/dashboard` |
| Ollama | `ollama.datalab.localhost` | only with `ollama.enabled` |
| S3 (SeaweedFS) | `s3.datalab.localhost` | keys in the `<release>-data-lab-s3` Secret |

## 🤖 AI Tools

### LLM (Ollama, or any OpenAI-compatible endpoint, or none)
The notebooks talk to one OpenAI-compatible API through `lab_utils`. Pick one with the chart's `llm` values:
- **Local Ollama** (`ollama.enabled=true`, GPU): defaults `gemma3:4b` (chat), `qwen3:4b-instruct` (tools) and `mxbai-embed-large` (embeddings).
- **Your own endpoint** (OpenRouter or any OpenAI-compatible server, no GPU): `llm.provider=openai`, `llm.baseUrl`, models, and the key in an existing Secret. Embeddings can use a different endpoint (`llm.embeddingBaseUrl`).
- **No AI** (`llm.provider=none`): every AI section of a notebook prints `skipped (no LLM configured): ...` and the rest runs.

### Vector Database (Qdrant)  
- **Purpose**: Store embeddings for RAG experiments
- **Web UI**: `http://localhost:6333/dashboard`
- **API**: `http://localhost:6333`

### Monitoring (Phoenix)
- **Purpose**: Basic AI operation tracing
- **Web UI**: `http://localhost:6006`

### Notebook Environment (Jupyter)
- **GenAI Environment**: Pre-installed packages
- **Core Libraries**: pandas, numpy, matplotlib, seaborn, plotly, scikit-learn
- **AI/ML Stack**: LangChain ecosystem, transformers, torch, sentence-transformers
- **Vector & Database**: qdrant-client, trino, sqlalchemy, boto3, s3fs
- **Document Processing**: pypdf2, python-docx, beautifulsoup4, tiktoken
- **Observability**: arize-phoenix, opentelemetry, openinference instrumentation
- **Development Tools**: ipywidgets, tqdm, rich, typer
- **dbt**: dbt-core 1.12.5 + dbt-trino 1.10.6 (project in `jupyter/notebooks/dbt/lakehouse_demo/`; the Jupyter image ships `git`, which `dbt debug`/`dbt deps` need)
- **Example notebooks**:
  - [`jupyter/notebooks/iceberg/`](jupyter/notebooks/iceberg/) — Iceberg 1.12 techniques on the Northwind shop: `00` what's new + live support matrix, `01` Hilbert vs Z-order on delivery points ("orders near Berlin"), `02` the clickstream as VARIANT (format v3) + funnel per product, `03` a customer's right to be forgotten (deletion vectors, row lineage, snapshot expiry), `04` the order stream into `shop_bronze.order_events` (streaming merge-append, Parquet tuning)
  - [`jupyter/notebooks/dbt/`](jupyter/notebooks/dbt/) — dbt (dbt-core 1.12.5 + dbt-trino 1.10.6 on Trino catalog `iceberg`): `01` getting started, `02` incremental merge + snapshots, `03` tests/contracts/unit tests. Project: `jupyter/notebooks/dbt/lakehouse_demo/`
  - [`jupyter/notebooks/semantic/`](jupyter/notebooks/semantic/) — `01` chat with your data, step by step: the dbt shop's seven tables (split payments, refunds in installments, costs and regions joins away), its Apache Ossie model, the Kubeflow Semantic Operator's MCP tools by hand, then a local LLM (`qwen3:4b-instruct`) answering through them vs writing raw SQL, scored against ground truth, plus row-level governance (Helm chart with `semantic.enabled` only)
  - `jupyter/notebooks/lab_data.py` — Northwind Gear, the one shop the notebooks share: clickstream, deliveries, order stream and product documents derived deterministically from the dbt seeds
  - `jupyter/notebooks/data_lab_playground.ipynb` (the tour: pipeline map, bronze snapshot, cross-catalog Trino join, rows per layer) and `rag_demo.ipynb` (why customers send products back: refund rate per product from the semantic layer, Trino or the seeds, plus support tickets retrieved from Qdrant)
  - [`jupyter/notebooks/ops/`](jupyter/notebooks/ops/) — the Airflow DAG `dags/northwind_daily.py` (Spark Connect ingest of the day's order events into `shop_bronze.order_events` → `dbt build` on Trino → quality checks → `shop_gold.product_kpis`, and a wait for the SemanticModel's `Published` condition; runs on the lab Airflow image `airflow/Dockerfile` with `airflow.labPipelines.enabled`) and `01_pipeline_agent.ipynb`, an LLM agent over the Airflow and Spark History MCP servers (Helm chart only; the workstation profile turns all of it on and mounts `ops/dags` as the DAG folder)
  - `jupyter/notebooks/lab_utils.py` — shared helper: every service endpoint (env set by the chart), `get_spark()` (Spark Connect), `trino()`, `dbt()`, `reset_dbt_demo()`, the semantic-layer chat helpers
- **Default Kernel**: GenAI Analytics (Python 3.12)
- **Access**: `http://localhost:8888` (password: 123456)

## 📊 Usage Examples

All notebooks run in the lab's Jupyter, where the chart sets every endpoint as an environment variable. Read them from
`lab_utils` instead of hardcoding hosts:

```python
import lab_utils as lu

spark = lu.get_spark()                       # Spark Connect; the server has the Iceberg catalog (type=rest)
lu.trino("SELECT * FROM iceberg.shop_bronze.order_events LIMIT 5")
lu.QDRANT_URL                                # also OLLAMA_URL, PHOENIX_URL, S3_ENDPOINT, ...
```

`type=hive` for the Iceberg catalog does not work with HMS 4.2.1 (it removed the Thrift `get_table` call), so every
client uses the HMS Iceberg REST catalog.

## 🛠️ Basic Configuration

- Credentials are never in values: each comes from an `existingSecret` or is generated once per release.
- Spark is Spark Connect on Kubernetes; executors are pods.
- Models of a local Ollama live on its persistent volume.

## 🔧 Platform Management

```bash
kubectl -n datalab get pods
kubectl -n datalab logs deploy/datalab-data-lab-trino
helm test datalab -n datalab                  # Spark writes an Iceberg table, Trino reads it back (+ the 19-check e2e)
hack/workstation/e2e.sh                       # the full workstation check
```

## 🧪 End-to-End Lakehouse Test

`helm test` and `hack/workstation/e2e.sh` run `charts/data-lab/files/e2e_lakehouse.py`: 19 checks of Spark → Iceberg via
HMS (create, insert, schema evolution, snapshots, time travel), Trino `iceberg` reads and writes (incl. `$snapshots` and
`FOR VERSION AS OF`), Spark reading Trino's writes, Spark Parquet → Trino `hive` external table, Trino `hive` CTAS, the
`lakehouse` catalog over both table types, a cross-catalog `hive` ⋈ `iceberg` JOIN, and that the objects on SeaweedFS
are plain Parquet (`PAR1` magic). No GPU needed. CI runs the same on kind (`hack/chart-e2e-kind.sh`).

## 🤝 Contributing

Found a bug or have an idea? Feel free to:
- Open GitHub issues for problems or suggestions
- Submit pull requests for improvements  
- Add example notebooks or documentation

## Acknowledgments

**Built with AI Assistance**

This project was developed with GitHub Copilot (powered by Claude Sonnet 4), demonstrating the power of human-AI partnership in creating comprehensive data platforms.