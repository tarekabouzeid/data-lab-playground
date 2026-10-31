# Testing the lab on your machine

A checklist for a manual run before a release. Each step has a pass criterion and a line saying what to paste back if it
fails. CI already covers lint, render tests and the kind scenarios. This list adds what CI cannot do: a GPU, a long-lived
minikube lab, a browser, and the notebooks.

Prerequisites: Docker, kind, minikube 1.39, Helm 4.x (3.19+ works for the chart), kubectl, jq, Python 3.11+ with
`pytest pyyaml jsonschema`, and an NVIDIA GPU with the container toolkit for step 4. Steps 1 to 3 and 5 need no GPU.

## 1. Static checks (no cluster)

```bash
python3 -m pytest tests/chart tests/unit -q
hack/check-compose-parity.py
for f in charts/data-lab/ci/*-values.yaml; do helm lint --strict charts/data-lab --kube-version 1.34.0 -f "$f" --set global.apiChecks=false; done
```

Pass: every test passes, `compose parity: 0 problem(s)`, and every profile lints.
If it fails, paste back the pytest summary or the failing lint output.

## 2. Docker Compose (frozen)

```bash
tests/e2e/run-e2e.sh --build --down
```

Pass: `E2E SUMMARY: 19/19 checks passed`.

Then run `./start-platform.sh`, open `semantic/01_chat_with_your_data.ipynb` in Jupyter (http://localhost:8888) and run the
first code cell. Pass: it stops with *"The semantic layer runs only on the Helm chart with semantic.enabled=true (Docker
Compose is frozen and does not include it)."* Run `data_lab_playground.ipynb` and `rag_demo.ipynb` too. Both now work on the Northwind shop (`lab_data.py`); on
Compose `rag_demo` takes its numbers from Trino over the dbt marts (after `dbt/01`) or from the seeds. Their service
URLs come from `lab_utils.py`, with the Compose hosts as defaults. `ops/01_pipeline_agent.ipynb` stops with *"The Airflow and
Spark History MCP servers run only on the Helm chart ..."*.
If it fails, paste back the e2e summary or the failing cell's traceback.

## 3. kind (the Sandpiper-style installs)

```bash
hack/kind-up.sh            # add --calico on WSL2 (kindnet cannot enforce NetworkPolicy there)
hack/chart-e2e-kind.sh minimal
hack/chart-e2e-kind.sh default
hack/chart-e2e-kind.sh parallel
hack/chart-e2e-kind.sh semantic
kind delete cluster --name datalab
```

Pass:
- every scenario ends with `=== PASS: <scenario> ===`
- `minimal` prints `minimal profile Ready in <N>s` with N under 180
- `semantic` prints:
  - `model published after the tables appeared`
  - `DAG wait_for_published (airflow ServiceAccount) times out while the model is unpublished`, then `DAG wait_for_published sees Published=True`
  - `total_revenue = 40.0`, then `refund_rate = 0.26 gross_margin = 20.0` and `dach_sales order_count = 3`
  - `role marketing is refused stg_customers.email`
  - `second semantic-enabled release refused (C14)`
  - `namespace dl-semantic is clean`

`hack/kind-up.sh` builds the lab images and loads every image the scenarios need, the two Semantic Operator images
built from upstream source by `hack/build-semantic-operator.sh` included.

If it fails, paste back the last 60 lines of the failing scenario. The script prints pod logs and events on failure.

## 4. Workstation lab (minikube + GPU)

```bash
hack/workstation/setup-minikube.sh --gpu-mode device-plugin     # or --gpu-mode none without a GPU
hack/workstation/build-images.sh                                 # --load if minikube is not on the docker runtime
hack/workstation/deploy.sh
hack/workstation/e2e.sh
```

Pass:
- `setup-minikube.sh` ends with `Cluster ready`, and prints `nvidia.com/gpu: 1` in GPU mode.
- `deploy.sh` ends with `Deployed.`. The SemanticModel line may say `Published=False` until the dbt tables exist; that is
  expected.
- `e2e.sh` prints all of these, then `workstation e2e passed`:
  - `helm test passed`
  - `SparkApplication: 19/19`
  - one `ok` line per notebook, 11 in total
  - `northwind_daily run e2e-<n>: success` (the DAG from `ops/dags`, triggered after the dbt notebooks: Spark Connect ingest,
    `dbt build`, checks, `shop_gold.product_kpis`, and the SemanticModel `Published`), then `ok ops/01_pipeline_agent.ipynb`
  - one `ok` line per gateway hostname

Then check by hand with `hack/workstation/gateway-forward.sh` running in another terminal:

| Check | Pass |
|---|---|
| `http://jupyter.datalab.localhost:8080` | JupyterLab opens. The token is `kubectl -n datalab get secret datalab-data-lab-jupyter -o jsonpath='{.data.token}' \| base64 -d`. Editing a notebook changes the file in your repo's `jupyter/notebooks/`. |
| `trino`, `airflow`, `spark`, `spark-history`, `phoenix`, `qdrant`, `ollama`, `s3` `.datalab.localhost:8080` | each answers. Trino and Airflow show their UIs; `s3` answers 403 to an anonymous request. |
| `http://headlamp.datalab.localhost:8080` | Headlamp logs in with the token from `hack/workstation/headlamp-token.sh`; the sidebar shows *Kubeflow → Spark*, listing the e2e SparkApplication. |
| GPU | `kubectl -n datalab exec deploy/datalab-data-lab-ollama -- nvidia-smi` lists the GPU. |
| `semantic/01_chat_with_your_data.ipynb` (run all) | the refund-rate cell shows the naive join and the semantic layer disagreeing (installments); the step-by-step cell ends with an answer whose numbers match the rows above it; the score table shows the semantic path at 6/6 (the raw text-to-SQL column is informational); `marketing` is refused the e-mail filter and `dach_sales` sees only DACH. |
| `rag_demo.ipynb` | `numbers from: semantic layer (certified metrics)`; the router answer names the Wi-Fi drop after firmware 2.4.1 and the SSD answer the dead-on-arrival drives, citing order ids (GPU: fast replies). |
| `data_lab_playground.ipynb` (after the other notebooks) | the *Rows per layer* table shows a count for every table except `shop_gold.product_kpis` until the DAG has run. |
| `http://airflow.datalab.localhost:8080` | DAG `northwind_daily` is listed, unpaused, and its e2e run is green: `ingest` returns a new `replay`, `build` logs `Done. PASS=50`, `published` succeeds; `shop_gold.product_kpis` exists in Trino. |
| `ops/01_pipeline_agent.ipynb` | the first answer names the latest `northwind_daily` run and its tasks; the steps table shows Airflow tool calls. |

If it fails, paste back the script's last lines, or for a UI the HTTP status from `curl -sI -H 'Host: <host>' http://127.0.0.1:8080/`.

## 5. Sandpiper portability

On a kind cluster from step 3 (`hack/kind-up.sh`):

```bash
# a) two default releases in one namespace, both healthy
helm install a charts/data-lab -n shared --create-namespace --wait --timeout 15m
helm install b charts/data-lab -n shared --wait --timeout 15m
helm test a -n shared && helm test b -n shared

# b) upgrading from chart 0.1.0 with default values creates and deletes nothing
git worktree add /tmp/chart-010 c18e8b7
helm install u /tmp/chart-010/charts/data-lab -n up --create-namespace --wait --timeout 15m
kubectl -n up get all,cm,secret,pvc,networkpolicy,role,rolebinding,sa -o name | sort > /tmp/before.txt
helm upgrade u charts/data-lab -n up --wait --timeout 15m
kubectl -n up get all,cm,secret,pvc,networkpolicy,role,rolebinding,sa -o name | sort > /tmp/after.txt
diff /tmp/before.txt /tmp/after.txt     # pod and ReplicaSet names may differ (template labels change); nothing else

# c) a second semantic-enabled release in one namespace is refused
kubectl apply -f hack/crds/semantic.ossie.io_semanticmodels.yaml
helm install s1 charts/data-lab -n sem --create-namespace -f charts/data-lab/ci/semantic-values.yaml --wait --timeout 15m
helm install s2 charts/data-lab -n sem -f charts/data-lab/ci/semantic-values.yaml   # must fail with "(C14)"
helm uninstall s1 -n sem && kubectl -n sem get semanticmodels   # must list nothing
```

Pass:
- a) both test runs pass.
- b) the diff shows only pod and ReplicaSet name changes.
- c) the second install fails with the C14 message, and after the uninstall no SemanticModel is left.

If it fails, paste back the failing command's output (for b, the diff).
