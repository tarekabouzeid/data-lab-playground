#!/usr/bin/env bash
# End-to-end test of the lakehouse core on Kubernetes: the same 19 checks as tests/e2e/run-e2e.sh
# (Spark x Trino x hive/iceberg/lakehouse catalogs + plain-Parquet storage checks). No GPU needed.
#
# Deploys SeaweedFS, metastore-db, Hive Metastore, Trino (+ the Spark Connect server for --mode connect) into its OWN
# namespace (datalab-e2e), so a running platform in `datalab` is not touched, and runs tests/e2e/e2e_lakehouse.py:
#   application  a SparkApplication: the Spark Operator spark-submits the script with our image   (default)
#   connect      a Job in the jupyter image that talks to the chart's Spark Connect server (the notebook path)
#   both         one after the other
#
# Usage: helm/scripts/e2e-k8s.sh [--mode application|connect|both] [--timeout 1800] [--keep]
# Prerequisites: helm/scripts/setup-minikube.sh --gpu-mode none (or any cluster with the Spark Operator watching
# datalab-e2e) and helm/scripts/build-images.sh.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODE=application; TIMEOUT=1800; KEEP=false
E2E_NAMESPACE="${E2E_NAMESPACE:-datalab-e2e}"; E2E_RELEASE=datalab-e2e
while [[ $# -gt 0 ]]; do
  case $1 in
    --mode) MODE=$2; shift 2;; --timeout) TIMEOUT=$2; shift 2;; --keep) KEEP=true; shift;;
    -h|--help) sed -n '2,15p' "$0"; exit 0;;
    *) die "unknown option: $1";;
  esac
done
[[ $MODE =~ ^(application|connect|both)$ ]] || die "--mode must be application, connect or both"
require helm; require kubectl

kubectl get namespace "$E2E_NAMESPACE" >/dev/null 2>&1 || {
  kubectl create namespace "$E2E_NAMESPACE"
  kubectl label namespace "$E2E_NAMESPACE" pod-security.kubernetes.io/warn=baseline --overwrite >/dev/null
  warn "created namespace $E2E_NAMESPACE; the Spark Operator must watch it (setup-minikube.sh does that: spark.jobNamespaces)"
}
K="kubectl -n $E2E_NAMESPACE"

deploy() {   # deploy <mode>
  local mode=$1 extra=()
  [[ $mode == application ]] && extra=(--set spark.enabled=false)
  log "Deploying the lakehouse core for mode=$mode into $E2E_NAMESPACE"
  helm upgrade --install "$E2E_RELEASE" "$CHART" -n "$E2E_NAMESPACE" --wait --timeout "${TIMEOUT}s" \
    --set jupyter.enabled=false --set phoenix.enabled=false --set qdrant.enabled=false \
    --set ollama.enabled=false --set gateway.enabled=false \
    --set e2e.enabled=true --set "e2e.mode=$mode" ${extra[@]+"${extra[@]}"}
  log "Waiting for the warehouse bucket"
  $K wait --for=condition=complete --timeout=300s job -l app.kubernetes.io/name=seaweedfs-bucket
}

summary_ok() { grep -q "E2E SUMMARY: 19/19 checks passed" <<<"$1"; }

run_application() {
  deploy application
  log "Waiting for SparkApplication/datalab-e2e"
  local deadline=$((SECONDS + TIMEOUT)) state=""
  while (( SECONDS < deadline )); do
    state=$($K get sparkapplication datalab-e2e -o jsonpath='{.status.applicationState.state}' 2>/dev/null || true)
    case $state in COMPLETED|FAILED|SUBMISSION_FAILED) break;; esac
    sleep 10
  done
  local logs; logs=$($K logs datalab-e2e-driver 2>&1 || true)
  echo "$logs" | grep -E '^\[(PASS|FAIL)\]|E2E SUMMARY|Error|Exception' | tail -40
  [[ $state == COMPLETED ]] || { $K describe sparkapplication datalab-e2e | tail -30; die "SparkApplication ended in state '${state:-none}'"; }
  summary_ok "$logs" || die "driver log has no '19/19 checks passed' line"
  ok "application mode: 19/19"
}

run_connect() {
  deploy connect
  log "Waiting for the Spark Connect server (SparkConnect/datalab-spark)"
  $K wait --for=jsonpath='{.status.state}'=Ready --timeout="${TIMEOUT}s" sparkconnect/datalab-spark
  log "Waiting for Job/datalab-e2e-connect"
  $K wait --for=condition=complete --timeout="${TIMEOUT}s" job/datalab-e2e-connect \
    || { $K logs job/datalab-e2e-connect | tail -40; die "connect-mode e2e Job did not complete"; }
  local logs; logs=$($K logs job/datalab-e2e-connect 2>&1)
  echo "$logs" | grep -E '^\[(PASS|FAIL)\]|E2E SUMMARY'
  summary_ok "$logs" || die "Job log has no '19/19 checks passed' line"
  ok "connect mode: 19/19"
}

cleanup() {
  [[ $KEEP == true ]] && { warn "--keep: leaving release $E2E_RELEASE in namespace $E2E_NAMESPACE"; return; }
  log "Cleaning up"
  helm uninstall "$E2E_RELEASE" -n "$E2E_NAMESPACE" >/dev/null 2>&1 || true
  $K delete pvc --all --wait=false >/dev/null 2>&1 || true   # StatefulSet PVCs outlive the release (retain policy)
}

rc=0
case $MODE in
  application) run_application || rc=$?;;
  connect)     run_connect || rc=$?;;
  both)        run_application || rc=$?
               if [[ $rc == 0 ]]; then
                 # a fresh start: the previous run leaves tables behind in the warehouse
                 KEEP=false cleanup; sleep 15; run_connect || rc=$?
               fi;;
esac
cleanup
exit $rc
