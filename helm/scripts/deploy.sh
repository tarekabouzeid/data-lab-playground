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
ok "Deployed. The ollama-pull Job keeps downloading models in the background: kubectl -n $NAMESPACE logs -f job/<ollama-pull-...>"
echo "Open the UIs: run \`minikube tunnel\` in another terminal, then helm/scripts/hosts.sh (or helm/scripts/port-forward.sh)."
