#!/usr/bin/env bash
# Install / upgrade Headlamp (official chart) with the Kubeflow plugin into the platform namespace.
# It is reachable at http://headlamp.<gateway.domain>/ once the datalab chart (Gateway + HTTPRoute) is deployed.
# Sign in with a token: helm/scripts/headlamp-token.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require helm; require kubectl
kubectl get namespace "$NAMESPACE" >/dev/null 2>&1 || die "namespace $NAMESPACE does not exist; run helm/scripts/setup-minikube.sh first"
log "Installing Headlamp $HEADLAMP_VERSION (+ Kubeflow plugin $HEADLAMP_KUBEFLOW_PLUGIN_VERSION) into $NAMESPACE"
helm repo add headlamp "$HEADLAMP_REPO" >/dev/null 2>&1 || true
helm repo update headlamp >/dev/null
helm upgrade --install headlamp headlamp/headlamp --version "$HEADLAMP_VERSION" \
  -n "$NAMESPACE" -f "$HELM_DIR/headlamp-values.yaml" --wait --timeout 10m
ok "Headlamp is running. The plugin is installed by the 'headlamp-plugin' sidecar: kubectl -n $NAMESPACE logs deploy/headlamp -c headlamp-plugin"
echo "Next: helm/scripts/headlamp-token.sh, then open http://headlamp.<gateway.domain>/ (helm/scripts/gateway-forward.sh)."
