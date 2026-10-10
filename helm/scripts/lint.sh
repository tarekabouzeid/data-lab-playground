#!/usr/bin/env bash
# Static checks, no cluster needed: helm lint, render every GPU/e2e/variant combination, and the compose-parity check.
# With --server-dry-run it also sends every render to the current cluster's API server (needs the Spark Operator and
# Gateway API CRDs installed), which validates the manifests against the real schemas.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require helm
DRY=false; [[ ${1:-} == --server-dry-run ]] && DRY=true

log "check-parity.sh"; "$HELM_DIR/scripts/check-parity.sh"

# `helm lint` renders without a cluster and cannot see CRDs, so the chart's capability checks are switched off for it
# only. `helm template` below keeps them on and answers them through --api-versions.
log "helm lint"
helm lint "$CHART" --set apiChecks=false
helm lint "$CHART" -f "$CHART/values-minikube.yaml" --set apiChecks=false

# name|extra helm arguments   (plain list: also works with the bash 3.2 that ships with macOS)
CASES=(
  "default|"
  "minikube|-f $CHART/values-minikube.yaml"
  "gpu-dra|--set gpu.mode=dra"
  "gpu-none|--set gpu.mode=none"
  "e2e-application|--set e2e.enabled=true --set spark.enabled=false"
  "e2e-connect|--set e2e.enabled=true --set e2e.mode=connect"
  "variants|--set jupyter.notebooks.mode=pvc --set spark.dynamicAllocation.enabled=true --set gateway.createClass=false"
  "core-only|--set jupyter.enabled=false --set phoenix.enabled=false --set qdrant.enabled=false --set ollama.enabled=false --set gateway.enabled=false"
)
for entry in "${CASES[@]}"; do
  name=${entry%%|*}; args=${entry#*|}
  out=$(mktemp)
  # shellcheck disable=SC2086
  helm template "$RELEASE" "$CHART" -n "$NAMESPACE" "${HELM_API_VERSIONS[@]}" $args > "$out" \
    || die "helm template failed for case '$name'"
  if [[ $DRY == true ]]; then
    kubectl apply --dry-run=server -f "$out" -n "$NAMESPACE" >/dev/null || die "server dry-run failed for case '$name'"
    ok "$name: rendered $(grep -c '^kind:' "$out") objects, server dry-run ok"
  else
    ok "$name: rendered $(grep -c '^kind:' "$out") objects"
  fi
  rm -f "$out"
done

log "the capability checks must fire when the APIs are missing"
if helm template "$RELEASE" "$CHART" -n "$NAMESPACE" >/dev/null 2>&1; then
  die "expected 'helm template' without --api-versions to fail (apiChecks)"
else
  ok "missing-CRD check fires"
fi
ok "all static checks passed"
