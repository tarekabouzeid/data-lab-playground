#!/usr/bin/env bash
# End-to-end test of charts/data-lab on an existing kind cluster (CI: .github/workflows/chart-e2e.yaml).
#
# Scenarios (run one at a time so a single kind node can hold them):
#   default   - chart defaults (hms, seaweedfs, spark on k8s executors, trino, airflow); helm test
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
    left=$(kubectl -n "$ns" get pods,pvc,secrets,configmaps,services,deployments,jobs,roles,rolebindings,serviceaccounts,networkpolicies \
      --no-headers -o name 2>/dev/null | grep -vE '^(configmap/kube-root-ca.crt|serviceaccount/default)$' || true)
    [ -z "$left" ] && { echo "namespace $ns is clean"; kubectl delete namespace "$ns" --wait=false >/dev/null; return 0; }
    sleep 5
  done
  echo "FAIL: leftovers in $ns after uninstall:"; echo "$left"; return 1
}

scenario_default() {
  log "default profile"
  install lab dl-default
  assert_contract lab dl-default
  assert_executors dl-default
  run_test lab dl-default
  assert_stable_secrets lab dl-default
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
  all) scenario_minimal; scenario_default; scenario_parallel ;;
  *) echo "unknown scenario $SCENARIO"; exit 2 ;;
esac
log "PASS: $SCENARIO"
