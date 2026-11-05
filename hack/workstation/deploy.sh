#!/usr/bin/env bash
# Install / upgrade the workstation lab: charts/data-lab with ci/workstation-values.yaml as release `datalab`.
# Extra arguments go to `helm upgrade --install`, e.g.
#   hack/workstation/deploy.sh --set ollama.gpu.dra.enabled=true     # GPU through DRA (setup-minikube.sh --gpu-mode dra)
#   hack/workstation/deploy.sh --set airflow.enabled=false            # lighter
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require helm; require kubectl; require jq
kubectl get namespace "$NAMESPACE" >/dev/null 2>&1 \
  || die "namespace $NAMESPACE does not exist; run hack/workstation/setup-minikube.sh first"

extra=()
gpus=$(kubectl get nodes -o json | jq -r '[.items[].status.capacity["nvidia.com/gpu"] // "0" | tonumber] | add')
if [[ ${gpus:-0} -lt 1 && " $* " != *"gpu.dra.enabled=true"* ]]; then
  warn "no nvidia.com/gpu on the node: Ollama runs on the CPU (slow); setup-minikube.sh --gpu-mode device-plugin adds a GPU"
  extra=(--set ollama.gpu.enabled=false)
fi

log "helm upgrade --install $RELEASE (namespace $NAMESPACE, profile ci/workstation-values.yaml)"
helm upgrade --install "$RELEASE" "$CHART" -n "$NAMESPACE" -f "$PROFILE" ${extra[@]+"${extra[@]}"} \
  --wait --timeout 25m "$@"

# helm --wait ignores custom resources. The dbt-demo model publishes once its tables exist (dbt build, e.g. the
# semantic/01 notebook does it), so only report its state here.
st=$(kubectl -n "$NAMESPACE" get semanticmodel "$FULLNAME-dbt-demo" \
  -o jsonpath='{.status.conditions[?(@.type=="Published")].status}' 2>/dev/null || true)
echo "SemanticModel $FULLNAME-dbt-demo: Published=${st:-<not reconciled yet>} (True after the dbt demo tables are built)"
ok "Deployed. Ollama pulls its models before it turns Ready (first start: several minutes)."
echo "Open the UIs: hack/workstation/gateway-forward.sh, then http://<route>.$RELEASE.localhost:8080/"
