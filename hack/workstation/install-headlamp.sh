#!/usr/bin/env bash
# Install / upgrade Headlamp (official chart) with the Kubeflow plugin into the lab namespace. It is its own release
# (it needs cluster-wide read access), reachable at http://headlamp.<release>.localhost:8080/ through the chart's
# gateway.extraRoutes once charts/data-lab is deployed with the workstation profile.
# Sign in with a token: hack/workstation/headlamp-token.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require helm; require kubectl
kubectl get namespace "$NAMESPACE" >/dev/null 2>&1 || die "namespace $NAMESPACE does not exist; run hack/workstation/setup-minikube.sh first"
log "Installing Headlamp $HEADLAMP_VERSION (+ Kubeflow plugin $HEADLAMP_KUBEFLOW_PLUGIN_VERSION) into $NAMESPACE"
helm repo add headlamp "$HEADLAMP_REPO" >/dev/null 2>&1 || true
helm repo update headlamp >/dev/null
helm upgrade --install headlamp headlamp/headlamp --version "$HEADLAMP_VERSION" \
  -n "$NAMESPACE" -f "$WS_DIR/headlamp-values.yaml" --wait --timeout 10m
ok "Headlamp is running. The plugin is installed by the 'headlamp-plugin' sidecar: kubectl -n $NAMESPACE logs deploy/headlamp -c headlamp-plugin"
echo "Next: hack/workstation/headlamp-token.sh, then open http://headlamp.$RELEASE.localhost:8080/ (hack/workstation/gateway-forward.sh)."
