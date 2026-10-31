#!/usr/bin/env bash
# End-to-end test of charts/data-lab on an existing kind cluster (CI: .github/workflows/chart-e2e.yaml).
#
# Scenarios (run one at a time so a single kind node can hold them):
#   default   - chart defaults (hms, seaweedfs, spark on k8s executors, trino, airflow); helm test;
#               a DAG delivered through the DAG PVC runs, and its task log is still readable after an Airflow restart;
#               Spark History Server shows the Connect app after a restart; both MCP servers answer (read-only enforced)
#   minimal   - ci/minimal-values.yaml; must be Ready in < MINIMAL_BUDGET_S seconds; helm test
#   parallel  - two releases in one namespace + the same release name in two namespaces (ci/small-values.yaml);
#               all four installed at once, helm test on all
# After each scenario: helm uninstall, then assert the namespace holds nothing but kube-root-ca.crt and the
# default ServiceAccount (no PVCs, no Secrets, no Spark executor leftovers).
#
# Usage: hack/chart-e2e-kind.sh [default|minimal|parallel|all]   (images must already be loaded into kind)
set -euo pipefail

CHART="$(cd "$(dirname "$0")/../charts/data-lab" && pwd)"
SCENARIO="${1:-all}"
MINIMAL_BUDGET_S="${MINIMAL_BUDGET_S:-180}"
TIMEOUT="${HELM_TIMEOUT:-15m}"

log() { echo -e "\n=== $* ==="; }

dump() {  # namespace
  echo "--- diagnostics for $1"
  kubectl -n "$1" get pods,pvc,svc -o wide || true
  kubectl -n "$1" get events --sort-by=.lastTimestamp | tail -40 || true
  for p in $(kubectl -n "$1" get pods -o name 2>/dev/null); do
    echo "--- logs $p"; kubectl -n "$1" logs "$p" --all-containers --tail=60 || true
  done
}

ensure_ns() { kubectl get namespace "$1" >/dev/null 2>&1 || kubectl create namespace "$1" >/dev/null; }

install() {  # release namespace [helm args...]   (namespace must exist when installs run in parallel)
  local rel=$1 ns=$2; shift 2
  ensure_ns "$ns"
  helm install "$rel" "$CHART" -n "$ns" --wait --timeout "$TIMEOUT" "$@" || { dump "$ns"; return 1; }
}

run_test() {  # release namespace
  # No --logs: hook-succeeded deletes the pod first. A failed pod is kept, so dump() shows its logs.
  helm test "$1" -n "$2" --timeout 10m || { dump "$2"; return 1; }
}

wait_all() {  # pids...: wait for every job, fail if any failed (an && chain would hide failures under set -e)
  local rc=0
  for pid in "$@"; do wait "$pid" || rc=1; done
  return $rc
}

assert_executors() {  # namespace: Spark (master=kubernetes) must have created executor pods
  local n
  n=$(kubectl -n "$1" get pods -l spark-role=executor --no-headers 2>/dev/null | wc -l)
  echo "spark executor pods in $1: $n"
  [ "$n" -ge 1 ] || { echo "FAIL: no Spark executor pods in $1"; return 1; }
}

assert_contract() {  # release namespace: endpoints ConfigMap lists Secrets that exist
  local cm="$1-data-lab-endpoints" json
  json=$(kubectl -n "$2" get configmap "$cm" -o jsonpath='{.data.ENDPOINTS_JSON}')
  for s in $(echo "$json" | python3 -c 'import json,sys
d=json.load(sys.stdin)
for v in d.values():
    if isinstance(v, dict) and "credentials" in v: print(v["credentials"]["secret"])'); do
    kubectl -n "$2" get secret "$s" >/dev/null || { echo "FAIL: endpoints ConfigMap names missing Secret $s"; return 1; }
  done
  echo "endpoints contract OK for $1"
}

assert_stable_secrets() {  # release namespace: generated credentials survive helm upgrade
  local before after
  before=$(kubectl -n "$2" get secrets -l app.kubernetes.io/instance="$1" -o jsonpath='{range .items[*]}{.metadata.name}{.data}{"\n"}{end}' | sha256sum)
  helm upgrade "$1" "$CHART" -n "$2" --reuse-values --wait --timeout "$TIMEOUT" >/dev/null
  after=$(kubectl -n "$2" get secrets -l app.kubernetes.io/instance="$1" -o jsonpath='{range .items[*]}{.metadata.name}{.data}{"\n"}{end}' | sha256sum)
  [ "$before" = "$after" ] || { echo "FAIL: generated Secrets changed on upgrade"; return 1; }
  echo "generated Secrets stable across upgrade for $1"
}

uninstall_and_assert_clean() {  # namespace releases...
  local ns=$1; shift
  for rel in "$@"; do helm uninstall "$rel" -n "$ns" --wait --timeout 5m; done
  # executor pods and Spark's executor ConfigMap are garbage-collected via ownerReferences
  for _ in $(seq 1 60); do
    left=$(kubectl -n "$ns" get pods,pvc,secrets,configmaps,services,deployments,jobs,roles,rolebindings,serviceaccounts,networkpolicies,leases \
      --no-headers -o name 2>/dev/null | grep -vE '^(configmap/kube-root-ca.crt|serviceaccount/default)$' || true)
    # SemanticModels (custom resources, only where the CRD exists) are listed separately: an unknown type would
    # make the combined get above fail and look clean.
    if kubectl get crd semanticmodels.semantic.ossie.io >/dev/null 2>&1; then
      left="$left$(kubectl -n "$ns" get semanticmodels.semantic.ossie.io --no-headers -o name 2>/dev/null || true)"
    fi
    [ -z "$left" ] && { echo "namespace $ns is clean"; kubectl delete namespace "$ns" --wait=false >/dev/null; return 0; }
    sleep 5
  done
  echo "FAIL: leftovers in $ns after uninstall:"; echo "$left"; return 1
}

assert_airflow_logs_persist() {  # release namespace: task logs stay readable in Airflow after the pod restarts
  local rel=$1 ns=$2 dep="$1-data-lab-airflow" run
  # deliver a DAG the way an external agent would: write it into the DAG PVC from another pod
  kubectl -n "$ns" apply -f - >/dev/null <<YAML
apiVersion: v1
kind: Pod
metadata: {name: dag-writer}
spec:
  restartPolicy: Never
  securityContext: {runAsNonRoot: true, runAsUser: 50000, runAsGroup: 0, seccompProfile: {type: RuntimeDefault}}
  containers:
    - name: w
      image: busybox:1.36
      securityContext: {allowPrivilegeEscalation: false, capabilities: {drop: [ALL]}}
      command: [sh, -c]
      args:
        - |
          cat > /dags/e2e_logs.py <<'PY'
          from datetime import datetime
          from airflow.sdk import dag, task

          @dag(dag_id="e2e_logs", schedule=None, start_date=datetime(2026, 1, 1), catchup=False)
          def e2e_logs():
              @task
              def hello():
                  print("E2E-LOG-MARKER")
              hello()

          e2e_logs()
          PY
      volumeMounts: [{name: dags, mountPath: /dags}]
  volumes: [{name: dags, persistentVolumeClaim: {claimName: $rel-data-lab-airflow-dags}}]
YAML
  kubectl -n "$ns" wait --for=jsonpath='{.status.phase}'=Succeeded pod/dag-writer --timeout=120s
  kubectl -n "$ns" delete pod dag-writer --wait=false >/dev/null
  run=$(kubectl -n "$ns" exec -i "deploy/$dep" -c api-server -- python - trigger e2e_logs hello E2E-LOG-MARKER \
          < "$(dirname "$0")/airflow-logs-check.py" | tee /dev/stderr | sed -n 's/^RUN_ID=//p') || true
  [ -n "$run" ] || { dump "$ns"; return 1; }
  kubectl -n "$ns" rollout restart "deploy/$dep" >/dev/null
  kubectl -n "$ns" rollout status "deploy/$dep" --timeout=10m
  kubectl -n "$ns" exec -i "deploy/$dep" -c api-server -- python - read e2e_logs "$run" hello E2E-LOG-MARKER \
    < "$(dirname "$0")/airflow-logs-check.py" || { dump "$ns"; return 1; }
  echo "Airflow task logs survive a pod restart for $rel"
}

assert_spark_history() {  # release namespace: a stopped Connect server shows up as a completed app with its jobs
  local rel=$1 ns=$2 apps=""
  # graceful stop closes the event log (S3 makes it visible only once closed)
  kubectl -n "$ns" rollout restart "deploy/$rel-data-lab-spark-connect" >/dev/null
  kubectl -n "$ns" rollout status "deploy/$rel-data-lab-spark-connect" --timeout=10m
  for _ in $(seq 1 40); do
    apps=$(kubectl -n "$ns" exec "deploy/$rel-data-lab-spark-history" -- curl -sf 'http://localhost:18080/api/v1/applications?status=completed' || true)
    echo "$apps" | grep -q '"completed" : true' && break
    sleep 5
  done
  echo "$apps" | grep -q '"completed" : true' || { echo "FAIL: no completed app in the Spark History Server"; dump "$ns"; return 1; }
  local app; app=$(echo "$apps" | python3 -c 'import json,sys; print(json.load(sys.stdin)[0]["id"])')
  local jobs; jobs=$(kubectl -n "$ns" exec "deploy/$rel-data-lab-spark-history" -- curl -sf "http://localhost:18080/api/v1/applications/$app/jobs" \
    | python3 -c 'import json,sys; print(len(json.load(sys.stdin)))')
  [ "$jobs" -ge 1 ] || { echo "FAIL: history app $app has no jobs"; return 1; }
  echo "Spark History Server shows completed app $app with $jobs jobs"
}

mcp_call() {  # release namespace url tool args: run hack/mcp-call.py inside a pod of the release (NetworkPolicy)
  kubectl -n "$2" exec -i "deploy/$1-data-lab-airflow" -c api-server -- python - "$3" "$4" "$5" < "$(dirname "$0")/mcp-call.py"
}

assert_mcps() {  # release namespace: both MCP servers answer real questions; Airflow rejects writes from the MCP user
  local rel=$1 ns=$2 out
  local shs="http://$rel-data-lab-spark-history-mcp.$ns.svc:18888/mcp" af="http://$rel-data-lab-airflow-mcp.$ns.svc:8000/mcp"
  out=$(mcp_call "$rel" "$ns" "$shs" list_applications '{"server": "history"}') || { echo "$out"; return 1; }
  echo "$out" | grep -q '"completed": true' || { echo "FAIL: spark MCP history: $out" | head -c 1500; return 1; }
  out=$(mcp_call "$rel" "$ns" "$shs" list_applications '{"server": "live"}') || { echo "$out"; return 1; }
  echo "$out" | grep -q 'Spark Connect server' || { echo "FAIL: spark MCP live: $out" | head -c 1500; return 1; }
  echo "Spark History MCP answers for history and live"
  out=$(mcp_call "$rel" "$ns" "$af" list_dags '{}') || { echo "$out"; return 1; }
  echo "$out" | grep -q 'e2e_logs' || { echo "FAIL: airflow MCP list_dags: $out" | head -c 1500; return 1; }
  out=$(mcp_call "$rel" "$ns" "$af" trigger_dag '{"dag_id": "e2e_logs"}') || { echo "$out"; return 1; }
  echo "$out" | grep -qiE 'read-only|403|forbidden' || { echo "FAIL: airflow MCP trigger_dag was not refused: $out" | head -c 1500; return 1; }
  echo "Airflow MCP lists DAGs and its trigger_dag is refused (read-only, viewer role)"
}

semantic_published() {  # namespace model: status of the Published condition ("" until reconciled)
  kubectl -n "$1" get semanticmodel "$2" -o jsonpath='{.status.conditions[?(@.type=="Published")].status}' 2>/dev/null
}

semantic_call() {  # release namespace role tool args: MCP call from the release's Spark Connect pod (NetworkPolicy)
  kubectl -n "$2" exec -i "deploy/$1-data-lab-spark-connect" -c spark-connect -- \
    env MCP_HEADERS="{\"X-Semantic-User\": \"e2e\", \"X-Semantic-Role\": \"$3\"}" \
    /opt/conda/bin/python - "http://$1-data-lab-semantic.$2.svc:8090/mcp" "$4" "$5" < "$(dirname "$0")/mcp-call.py"
}

dag_wait_published() {  # release namespace timeout: northwind_daily's wait_for_published, run in the Airflow
  # scheduler as the chart's airflow ServiceAccount (airflow.labPipelines + semantic: read-only SemanticModels)
  local dag; dag="$(dirname "$0")/../jupyter/notebooks/ops/dags/northwind_daily.py"
  kubectl -n "$2" exec -i "deploy/$1-data-lab-airflow" -c scheduler -- sh -c 'cat > /tmp/northwind_daily.py' < "$dag"
  kubectl -n "$2" exec "deploy/$1-data-lab-airflow" -c scheduler -- python -c "
import importlib.util
spec = importlib.util.spec_from_file_location('northwind_daily', '/tmp/northwind_daily.py')
dag = importlib.util.module_from_spec(spec); spec.loader.exec_module(dag)
print(dag.wait_for_published(timeout_s=$3, poll_s=5))"
}

trino_exec() {  # release namespace sql
  kubectl -n "$2" exec "deploy/$1-data-lab-trino" -- trino --execute "$3"
}

scenario_semantic() {
  log "semantic layer: Kubeflow Semantic Operator on the release's Trino"
  local rel=lab ns=dl-semantic model=lab-data-lab-dbt-demo st out
  kubectl apply -f "$(dirname "$0")/crds/semantic.ossie.io_semanticmodels.yaml"
  # Airflow with labPipelines: the northwind_daily DAG's wait_for_published runs against the real operator below
  install "$rel" "$ns" -f "$CHART/ci/semantic-values.yaml" --set airflow.enabled=true,airflow.labPipelines.enabled=true
  # Applied before its tables exist: the manager must reach Trino, find the tables missing (DriftDetected=True) and
  # not publish. Waiting for that also proves the manager is reconciling (an empty status would not).
  local drift="" deadline0=$((SECONDS + 180))
  until [ "$drift" = "True" ]; do
    [ $SECONDS -lt $deadline0 ] || { kubectl -n "$ns" get semanticmodel "$model" -o yaml; dump "$ns"; echo "FAIL: $model never reported DriftDetected=True before its tables existed"; return 1; }
    sleep 5
    drift=$(kubectl -n "$ns" get semanticmodel "$model" -o jsonpath='{.status.conditions[?(@.type=="DriftDetected")].status}' 2>/dev/null || true)
  done
  st=$(semantic_published "$ns" "$model")
  [ "$st" != "True" ] || { echo "FAIL: $model published before its tables exist"; return 1; }
  echo "model before data: DriftDetected=True, Published='${st}'"
  if out=$(dag_wait_published "$rel" "$ns" 0 2>&1); then
    echo "FAIL: northwind_daily's wait_for_published returned before the tables exist: $out"; return 1
  fi
  echo "$out" | grep -q "DriftDetected\|Published=" || { echo "FAIL: wait_for_published failed for another reason: $out" | tail -20; return 1; }
  echo "DAG wait_for_published (airflow ServiceAccount) times out while the model is unpublished"
  # The dbt demo's seven tables, small: 4 orders (one returned) -> total_revenue = 20 + 15 + 5 = 40.
  # Order 2 is refunded in two installments (5 + 5) and order 3 gets a 3.0 goodwill refund: refunded_amount = 13,
  # gross sales = 20 + 10 + 15 + 5 = 50, so refund_rate = 13 / 50 = 0.26 (a join of refunds to orders would count
  # order 2's amount twice: the planner must not).
  trino_exec "$rel" "$ns" "CREATE SCHEMA IF NOT EXISTS iceberg.dbt_demo"
  trino_exec "$rel" "$ns" "CREATE TABLE iceberg.dbt_demo.stg_customers AS SELECT * FROM (VALUES
    (1, 'Ann', 'ann@example.com', 'DE', 'pro', DATE '2026-01-02'),
    (2, 'Bob', 'bob@example.com', 'FR', 'free', DATE '2026-02-03'),
    (3, 'Cy', 'cy@example.com', 'DE', 'free', DATE '2026-03-04'))
    AS t(customer_id, name, email, country, plan, signup_date)"
  trino_exec "$rel" "$ns" "CREATE TABLE iceberg.dbt_demo.dim_regions AS SELECT * FROM (VALUES
    ('DE', 'DACH', 'Germany'), ('FR', 'Western Europe', 'France')) AS t(country, sales_region, country_name)"
  trino_exec "$rel" "$ns" "CREATE TABLE iceberg.dbt_demo.dim_products AS SELECT * FROM (VALUES
    (1, 'Lamp', 'home', 10.0), (2, 'Pen', 'office', 5.0)) AS t(product_id, product_name, category, unit_price)"
  trino_exec "$rel" "$ns" "CREATE TABLE iceberg.dbt_demo.dim_product_costs AS SELECT * FROM (VALUES
    (1, 6.0, 'Northwind'), (2, 2.0, 'Contoso')) AS t(product_id, unit_cost, supplier)"
  trino_exec "$rel" "$ns" "CREATE TABLE iceberg.dbt_demo.fct_orders AS SELECT * FROM (VALUES
    (1, 1, 1, 2, 10.0, 20.0, TIMESTAMP '2026-10-01 10:00:00', 'fulfilled'),
    (2, 2, 1, 1, 10.0, 10.0, TIMESTAMP '2026-10-02 10:00:00', 'returned'),
    (3, 1, 2, 3, 5.0, 15.0, TIMESTAMP '2026-10-03 10:00:00', 'fulfilled'),
    (4, 3, 2, 1, 5.0, 5.0, TIMESTAMP '2026-10-04 10:00:00', 'open'))
    AS t(order_id, customer_id, product_id, quantity, unit_price, amount, order_ts, status)"
  trino_exec "$rel" "$ns" "CREATE TABLE iceberg.dbt_demo.fct_payments AS SELECT * FROM (VALUES
    (1, 1, 'card', 20.0, TIMESTAMP '2026-10-01 10:01:00'), (2, 2, 'card', 10.0, TIMESTAMP '2026-10-02 10:01:00'),
    (3, 3, 'gift_card', 5.0, TIMESTAMP '2026-10-03 10:01:00'), (4, 3, 'card', 10.0, TIMESTAMP '2026-10-03 10:02:00'),
    (5, 4, 'paypal', 5.0, TIMESTAMP '2026-10-04 10:01:00')) AS t(payment_id, order_id, method, amount, paid_at)"
  trino_exec "$rel" "$ns" "CREATE TABLE iceberg.dbt_demo.fct_refunds AS SELECT * FROM (VALUES
    (1, 2, 5.0, TIMESTAMP '2026-10-05 10:00:00', 'returned'), (2, 2, 5.0, TIMESTAMP '2026-10-09 10:00:00', 'returned'),
    (3, 3, 3.0, TIMESTAMP '2026-10-06 10:00:00', 'damaged')) AS t(refund_id, order_id, amount, refunded_at, reason)"
  local deadline=$((SECONDS + 120))
  until [ "$(semantic_published "$ns" "$model")" = "True" ]; do
    [ $SECONDS -lt $deadline ] || { kubectl -n "$ns" get semanticmodel "$model" -o yaml; dump "$ns"; echo "FAIL: $model not published 120 s after its tables appeared"; return 1; }
    sleep 5
  done
  echo "model published after the tables appeared"
  out=$(dag_wait_published "$rel" "$ns" 60 2>&1) || { echo "$out" | tail -20; echo "FAIL: northwind_daily's wait_for_published"; return 1; }
  echo "$out" | grep -q "'published': True" || { echo "FAIL: wait_for_published: $out" | tail -5; return 1; }
  echo "DAG wait_for_published sees Published=True: $(echo "$out" | tail -1)"
  out=$(semantic_call "$rel" "$ns" analyst query_metric '{"model": "dbt_demo", "metrics": ["total_revenue"]}') || { echo "$out"; return 1; }
  echo "$out" | python3 -c 'import json,sys; r=json.load(sys.stdin); v=float(r["rows"][0][0]); print("total_revenue =", v, "| sql:", " ".join(r["sql"].split())); sys.exit(0 if abs(v-40) < 1e-6 else 1)' \
    || { echo "FAIL: total_revenue is not 40 (returned orders must be excluded): $out" | head -c 2000; return 1; }
  out=$(semantic_call "$rel" "$ns" marketing query_metric \
    '{"model": "dbt_demo", "metrics": ["order_count"], "filters": [{"field": "stg_customers.email", "op": "=", "value": "ann@example.com"}]}') || true
  echo "$out" | grep -qiE 'den(y|ied)|not (permitted|allowed|visible)|forbidden|unauthori|governance' \
    || { echo "FAIL: role marketing could filter on stg_customers.email: $out" | head -c 2000; return 1; }
  echo "role marketing is refused stg_customers.email"
  out=$(semantic_call "$rel" "$ns" analyst query_metric '{"model": "dbt_demo", "metrics": ["refund_rate", "gross_margin"]}') || { echo "$out"; return 1; }
  echo "$out" | python3 -c 'import json,sys; r=json.load(sys.stdin); rate, margin = (float(v) for v in r["rows"][0]); print("refund_rate =", rate, "gross_margin =", margin); sys.exit(0 if abs(rate-0.26) < 1e-6 and abs(margin-20) < 1e-6 else 1)' \
    || { echo "FAIL: refund_rate must be 0.26 (fan-out safe) and gross_margin 20 (= 20-12 + 15-6 + 5-2): $out" | head -c 2000; return 1; }
  out=$(semantic_call "$rel" "$ns" dach_sales query_metric '{"model": "dbt_demo", "metrics": ["order_count"]}') || { echo "$out"; return 1; }
  echo "$out" | python3 -c 'import json,sys; r=json.load(sys.stdin); v=int(r["rows"][0][0]); print("dach_sales order_count =", v); sys.exit(0 if v == 3 else 1)' \
    || { echo "FAIL: the dach_sales row filter must keep only DE orders (3 of 4): $out" | head -c 2000; return 1; }
  # C14: a second semantic-enabled release in the same namespace is refused at install time
  if out=$(helm install lab2 "$CHART" -n "$ns" -f "$CHART/ci/semantic-values.yaml" 2>&1); then
    helm uninstall lab2 -n "$ns" >/dev/null 2>&1 || true
    echo "FAIL: a second semantic-enabled release installed in $ns"; return 1
  fi
  echo "$out" | grep -q "C14" || { echo "FAIL: second release failed for another reason: $out"; return 1; }
  echo "second semantic-enabled release refused (C14)"
  run_test "$rel" "$ns"
  # The pre-delete hook removes the models while the manager still runs, so nothing is left behind.
  uninstall_and_assert_clean "$ns" "$rel"
}

scenario_default() {
  log "default profile"
  install lab dl-default --set sparkHistoryMcp.enabled=true,airflowMcp.enabled=true
  assert_contract lab dl-default
  assert_executors dl-default
  run_test lab dl-default
  assert_stable_secrets lab dl-default
  assert_airflow_logs_persist lab dl-default
  assert_spark_history lab dl-default
  assert_mcps lab dl-default
  uninstall_and_assert_clean dl-default lab
}

scenario_minimal() {
  log "minimal profile (budget ${MINIMAL_BUDGET_S}s)"
  local start=$SECONDS
  install lab dl-minimal -f "$CHART/ci/minimal-values.yaml"
  local took=$((SECONDS - start))
  echo "minimal profile Ready in ${took}s"
  echo "minimal_ready_seconds=${took}" >> "${GITHUB_OUTPUT:-/dev/null}"
  assert_executors dl-minimal
  run_test lab dl-minimal
  uninstall_and_assert_clean dl-minimal lab
  [ "$took" -lt "$MINIMAL_BUDGET_S" ] || { echo "FAIL: minimal profile took ${took}s (budget ${MINIMAL_BUDGET_S}s)"; return 1; }
}

scenario_parallel() {
  log "parallel: p1+p2 in dl-shared, 'lab' in dl-ns1 and dl-ns2"
  local small=(-f "$CHART/ci/small-values.yaml")
  for ns in dl-shared dl-ns1 dl-ns2; do ensure_ns "$ns"; done
  install p1 dl-shared "${small[@]}" & a=$!
  install p2 dl-shared "${small[@]}" & b=$!
  install lab dl-ns1 "${small[@]}" & c=$!
  install lab dl-ns2 "${small[@]}" & d=$!
  wait_all $a $b $c $d
  run_test p1 dl-shared & a=$!
  run_test p2 dl-shared & b=$!
  run_test lab dl-ns1 & c=$!
  run_test lab dl-ns2 & d=$!
  wait_all $a $b $c $d
  # NetworkPolicy: another release's pod must not reach this release's Trino (only if the CNI enforces policies)
  if kubectl -n dl-shared run np-probe --rm -i --restart=Never --image=busybox:1.36 \
      --labels=app.kubernetes.io/name=data-lab,app.kubernetes.io/instance=p2 \
      --overrides='{"spec":{"securityContext":{"runAsNonRoot":true,"runAsUser":65534,"seccompProfile":{"type":"RuntimeDefault"}},"containers":[{"name":"np-probe","image":"busybox:1.36","command":["wget","-q","-T","5","-O-","http://p1-data-lab-trino:8080/v1/info"],"securityContext":{"allowPrivilegeEscalation":false,"capabilities":{"drop":["ALL"]}}}]}}' \
      >/dev/null 2>&1; then
    echo "WARN: release p2 reached p1's Trino: the CNI does not enforce NetworkPolicy"
  else
    echo "NetworkPolicy isolates p1 from p2"
  fi
  uninstall_and_assert_clean dl-shared p1 p2
  uninstall_and_assert_clean dl-ns1 lab
  uninstall_and_assert_clean dl-ns2 lab
}

case "$SCENARIO" in
  default) scenario_default ;;
  minimal) scenario_minimal ;;
  parallel) scenario_parallel ;;
  semantic) scenario_semantic ;;
  all) scenario_minimal; scenario_default; scenario_parallel; scenario_semantic ;;
  *) echo "unknown scenario $SCENARIO"; exit 2 ;;
esac
log "PASS: $SCENARIO"
