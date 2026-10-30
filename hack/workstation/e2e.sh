#!/usr/bin/env bash
# End-to-end checks of the deployed workstation lab (hack/workstation/deploy.sh first). No GPU needed except for
# the GenAI notebooks.
#   1. helm test: Iceberg round trip + the 19-check lakehouse e2e (tests/e2e/e2e_lakehouse.py) over Spark Connect
#   2. the same 19 checks as a SparkApplication (Kubeflow Spark Operator spark-submits it with our Spark image)
#   3. every notebook, headless, inside the Jupyter pod (dbt before semantic/01)
#   4. every gateway hostname answers (through the Envoy proxy)
#
# Usage: hack/workstation/e2e.sh [--skip helm-test,spark-app,notebooks,gateway] [--timeout 1800]
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

SKIP=""; TIMEOUT=1800
while [[ $# -gt 0 ]]; do
  case $1 in
    --skip) SKIP=$2; shift 2;; --timeout) TIMEOUT=$2; shift 2;;
    -h|--help) sed -n '2,10p' "$0"; exit 0;;
    *) die "unknown option: $1";;
  esac
done
require helm; require kubectl; require curl
K="kubectl -n $NAMESPACE"
skip() { [[ ",$SKIP," == *",$1,"* ]]; }
endpoint() { $K get configmap "$FULLNAME-endpoints" -o jsonpath="{.data.$1}"; }
helm -n "$NAMESPACE" status "$RELEASE" >/dev/null 2>&1 || die "release $RELEASE is not deployed: hack/workstation/deploy.sh"
rc=0

if ! skip helm-test; then
  log "1/4 helm test (Iceberg round trip + 19-check lakehouse e2e over Spark Connect)"
  if helm test "$RELEASE" -n "$NAMESPACE" --timeout "${TIMEOUT}s"; then ok "helm test passed"
  else rc=1; warn "helm test failed: the failed test pod is kept, see $K logs pod/$FULLNAME-test-lakehouse"; fi
fi

if ! skip spark-app; then
  log "2/4 the lakehouse e2e as a SparkApplication (Spark Operator)"
  sa=$(endpoint SPARK_APPLICATION_SERVICE_ACCOUNT)
  [[ -n $sa ]] || die "spark.applications.enabled is off in this release"
  s3_key=$($K get secret "$FULLNAME-s3" -o jsonpath='{.data.access-key}' | base64 -d)
  s3_secret=$($K get secret "$FULLNAME-s3" -o jsonpath='{.data.secret-key}' | base64 -d)
  $K create configmap "$FULLNAME-e2e-script" --from-file="$ROOT/tests/e2e/e2e_lakehouse.py" --dry-run=client -o yaml | kubectl apply -f - >/dev/null
  $K delete sparkapplication "$FULLNAME-e2e" --ignore-not-found --wait >/dev/null
  image="ghcr.io/tarekabouzeid/data-lab-playground/spark:$APP_VERSION"
  env_block="
      - {name: AWS_REGION, value: us-east-1}
      - {name: TRINO_URL, value: \"$(endpoint TRINO_URL)\"}
      - {name: ICEBERG_REST_URI, value: \"$(endpoint ICEBERG_REST_URI)\"}
      - {name: HMS_THRIFT_URI, value: \"$(endpoint HMS_THRIFT_URI)\"}
      - {name: E2E_WAREHOUSE_BUCKET, value: \"$(endpoint S3_BUCKET)\"}"
  kubectl apply -f - <<YAML
apiVersion: sparkoperator.k8s.io/v1beta2
kind: SparkApplication
metadata:
  name: $FULLNAME-e2e
  namespace: $NAMESPACE
  labels: {app.kubernetes.io/name: data-lab, app.kubernetes.io/instance: $RELEASE, app.kubernetes.io/component: e2e}
spec:
  type: Python
  mode: cluster
  sparkVersion: "4.1.3"
  image: $image
  imagePullPolicy: IfNotPresent
  mainApplicationFile: local:///e2e/e2e_lakehouse.py
  restartPolicy: {type: Never}
  sparkConf:
    spark.hadoop.fs.s3a.endpoint: "$(endpoint S3_ENDPOINT)"
    spark.hadoop.fs.s3a.access.key: "$s3_key"
    spark.hadoop.fs.s3a.secret.key: "$s3_secret"
    spark.hadoop.fs.s3a.path.style.access: "true"
    spark.hadoop.fs.s3a.connection.ssl.enabled: "false"
    spark.hadoop.fs.s3a.aws.credentials.provider: org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider
  volumes:
    - name: e2e
      configMap: {name: $FULLNAME-e2e-script}
  driver:
    cores: 1
    memory: 1g
    serviceAccount: $sa
    labels: {app.kubernetes.io/name: data-lab, app.kubernetes.io/instance: $RELEASE}
    securityContext: {runAsNonRoot: true, runAsUser: 185, allowPrivilegeEscalation: false, capabilities: {drop: [ALL]}}
    env:$env_block
    volumeMounts:
      - {name: e2e, mountPath: /e2e}
  executor:
    instances: 1
    cores: 1
    memory: 1g
    labels: {app.kubernetes.io/name: data-lab, app.kubernetes.io/instance: $RELEASE}
    securityContext: {runAsNonRoot: true, runAsUser: 185, allowPrivilegeEscalation: false, capabilities: {drop: [ALL]}}
    env:$env_block
YAML
  deadline=$((SECONDS + TIMEOUT)); state=""
  while (( SECONDS < deadline )); do
    state=$($K get sparkapplication "$FULLNAME-e2e" -o jsonpath='{.status.applicationState.state}' 2>/dev/null || true)
    case $state in COMPLETED|FAILED|SUBMISSION_FAILED) break;; esac
    sleep 10
  done
  logs=$($K logs "$FULLNAME-e2e-driver" 2>&1 || true)
  echo "$logs" | grep -E '^\[(PASS|FAIL)\]|E2E SUMMARY' | tail -25
  if [[ $state == COMPLETED ]] && grep -q "E2E SUMMARY: 19/19 checks passed" <<<"$logs"; then ok "SparkApplication: 19/19"
  else rc=1; warn "SparkApplication ended in state '${state:-none}': $K describe sparkapplication $FULLNAME-e2e"; fi
fi

if ! skip notebooks; then
  log "3/4 notebooks, headless in the Jupyter pod"
  NOTEBOOKS=(data_lab_playground.ipynb rag_demo.ipynb iceberg/00_whats_new_and_whats_supported.ipynb
             iceberg/01_hilbert_clustering.ipynb iceberg/02_variant_semi_structured.ipynb
             iceberg/03_deletion_vectors_and_row_lineage.ipynb iceberg/04_streaming_and_parquet_tuning.ipynb
             dbt/01_getting_started.ipynb dbt/02_incremental_and_snapshots.ipynb dbt/03_tests_contracts_and_unit_tests.ipynb
             semantic/01_chat_with_your_data.ipynb)
  for nb in "${NOTEBOOKS[@]}"; do
    dir=$(dirname "$nb")
    if $K exec "deploy/$FULLNAME-jupyter" -- bash -c "cd /home/jovyan/work/$dir && jupyter nbconvert --to notebook --execute \
         --ExecutePreprocessor.timeout=900 --output /tmp/$(basename "$nb") $(basename "$nb")" >/tmp/nb.log 2>&1; then
      ok "$nb"
    else
      rc=1; warn "$nb failed:"; tail -15 /tmp/nb.log >&2
    fi
  done
fi

if ! skip gateway; then
  log "4/4 gateway hostnames"
  svc=$(kubectl get svc -A -l "gateway.envoyproxy.io/owning-gateway-name=$FULLNAME-gateway,gateway.envoyproxy.io/owning-gateway-namespace=$NAMESPACE" \
    -o jsonpath='{.items[0].metadata.namespace}/{.items[0].metadata.name}' 2>/dev/null || true)
  if [[ $svc == */* ]]; then
    kubectl -n "${svc%%/*}" port-forward "svc/${svc##*/}" 18080:80 >/dev/null 2>&1 & pf=$!
    trap 'kill $pf 2>/dev/null' EXIT; sleep 3
    for h in $($K get httproute -o jsonpath='{range .items[*]}{.spec.hostnames[*]}{" "}{end}'); do
      code=$(curl -s -o /dev/null -w '%{http_code}' -H "Host: $h" http://127.0.0.1:18080/ || echo 000)
      if [[ $code =~ ^[1-4] ]]; then ok "$h -> $code"; else rc=1; warn "$h -> $code"; fi
    done
  else
    rc=1; warn "no Envoy proxy Service for Gateway $NAMESPACE/$FULLNAME-gateway"
  fi
fi

[[ $rc == 0 ]] && ok "workstation e2e passed" || die "workstation e2e: failures above"
