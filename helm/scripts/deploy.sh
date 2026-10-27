#!/usr/bin/env bash
# Install / upgrade the platform chart. Extra arguments go to `helm upgrade --install`.
#   helm/scripts/deploy.sh                       # values.yaml + values-minikube.yaml
#   helm/scripts/deploy.sh --set gpu.mode=dra    # any chart value
#   helm/scripts/deploy.sh --set gpu.mode=none --set ollama.enabled=false
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require helm; require kubectl
kubectl get namespace "$NAMESPACE" >/dev/null 2>&1 \
  || die "namespace $NAMESPACE does not exist; run helm/scripts/setup-minikube.sh first"
log "helm upgrade --install $RELEASE (namespace $NAMESPACE)"
helm upgrade --install "$RELEASE" "$CHART" -n "$NAMESPACE" \
  -f "$CHART/values-minikube.yaml" --wait --timeout 25m "$@"
# helm --wait ignores custom resources: the Spark Connect server only exists once the operator reconciled it
if kubectl -n "$NAMESPACE" get sparkconnect datalab-spark >/dev/null 2>&1; then
  log "Waiting for SparkConnect/datalab-spark (started by the Spark Operator)"
  kubectl -n "$NAMESPACE" wait --for=jsonpath='{.status.state}'=Ready sparkconnect/datalab-spark --timeout=10m \
    || die "SparkConnect/datalab-spark is not Ready: kubectl -n spark-operator logs deploy/spark-operator-controller"
fi
ok "Deployed. The ollama-pull Job keeps downloading models in the background: kubectl -n $NAMESPACE logs -f job/<ollama-pull-...>"
echo "Open the UIs: helm/scripts/gateway-forward.sh, then http://<name>.datalab.localhost:8080/ (or \`minikube tunnel\` + helm/scripts/hosts.sh)."
