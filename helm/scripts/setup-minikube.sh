#!/usr/bin/env bash
# Create a local minikube cluster (optionally with an NVIDIA GPU) and install the platform prerequisites:
# Kubeflow Spark Operator, Envoy Gateway (Gateway API) and, optionally, the NVIDIA DRA driver.
# GPU steps follow https://minikube.sigs.k8s.io/docs/tutorials/nvidia/ (docker driver, Linux only).
#
# Usage: helm/scripts/setup-minikube.sh [--cpus 8] [--memory 24g] [--disk-size 80g]
#                                       [--gpu-mode device-plugin|dra|none] [--profile minikube] [--no-headlamp]
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

CPUS=8; MEMORY=24g; DISK=80g; GPU_MODE=device-plugin; HEADLAMP=true
while [[ $# -gt 0 ]]; do
  case $1 in
    --cpus) CPUS=$2; shift 2;;
    --memory) MEMORY=$2; shift 2;;
    --disk-size) DISK=$2; shift 2;;
    --gpu-mode) GPU_MODE=$2; shift 2;;
    --profile) MINIKUBE_PROFILE=$2; shift 2;;
    --no-headlamp) HEADLAMP=false; shift;;
    -h|--help) sed -n '2,9p' "$0"; exit 0;;
    *) die "unknown option: $1";;
  esac
done
[[ $GPU_MODE =~ ^(device-plugin|dra|none)$ ]] || die "--gpu-mode must be device-plugin, dra or none"

log "Checking tools"
for t in docker minikube kubectl helm jq; do require "$t"; done
helm_major=$(helm version --template '{{.Version}}' | sed -E 's/^v([0-9]+).*/\1/')
(( helm_major >= HELM_MIN_MAJOR )) || warn "helm v$helm_major found; the chart was developed with Helm v$HELM_MIN_MAJOR"
minikube version --short | grep -q "$MINIKUBE_VERSION" || warn "minikube $(minikube version --short) found; this setup was written for v$MINIKUBE_VERSION"

GPU_ARGS=()
if [[ $GPU_MODE != none ]]; then
  log "Checking the NVIDIA prerequisites (minikube docs: Using NVIDIA GPUs with minikube, docker driver)"
  [[ $(uname -s) == Linux ]] || die "NVIDIA GPUs in minikube are Linux-only"
  nvidia-smi >/dev/null 2>&1 || die "nvidia-smi failed: install the NVIDIA driver first"
  if [[ $(sysctl -n net.core.bpf_jit_harden 2>/dev/null || echo 0) != 0 ]]; then
    die 'net.core.bpf_jit_harden is not 0. Run: echo "net.core.bpf_jit_harden=0" | sudo tee -a /etc/sysctl.conf && sudo sysctl -p'
  fi
  docker info --format '{{json .Runtimes}}' | grep -q nvidia \
    || die "Docker has no nvidia runtime. Install the NVIDIA Container Toolkit, then: sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
  GPU_ARGS=(--gpus "${MINIKUBE_GPUS:-all}")      # docs also allow --gpus nvidia.com (CDI)
fi

log "Starting minikube $MINIKUBE_VERSION / Kubernetes $KUBERNETES_VERSION (profile $MINIKUBE_PROFILE)"
# --container-runtime docker: required by the GPU docs, and what makes `eval $(minikube docker-env)` builds work.
# The mount makes jupyter/notebooks editable from the host (chart: jupyter.notebooks.mode=hostPath).
minikube start -p "$MINIKUBE_PROFILE" --driver docker --container-runtime docker \
  --kubernetes-version "$KUBERNETES_VERSION" --cpus "$CPUS" --memory "$MEMORY" --disk-size "$DISK" \
  ${GPU_ARGS[@]+"${GPU_ARGS[@]}"} \
  --mount --mount-string "$ROOT/jupyter/notebooks:/datalab/notebooks"
kubectl config use-context "$MINIKUBE_PROFILE" >/dev/null

# datalab = the platform; datalab-e2e = where e2e-k8s.sh runs the lakehouse e2e without touching the platform
E2E_NAMESPACE="${E2E_NAMESPACE:-datalab-e2e}"
for ns in "$NAMESPACE" "$E2E_NAMESPACE"; do
  log "Creating namespace $ns (Pod Security: warn/audit at baseline)"
  kubectl create namespace "$ns" --dry-run=client -o yaml | kubectl apply -f - >/dev/null
  kubectl label namespace "$ns" --overwrite \
    pod-security.kubernetes.io/warn=baseline pod-security.kubernetes.io/audit=baseline >/dev/null
done

if [[ $GPU_MODE == device-plugin ]]; then
  log "Checking that the node advertises nvidia.com/gpu"
  for _ in $(seq 1 30); do
    n=$(kubectl get nodes -o json | jq -r '[.items[].status.capacity["nvidia.com/gpu"] // "0" | tonumber] | add')
    [[ $n -ge 1 ]] && break
    [[ ${tried:-} ]] || { minikube -p "$MINIKUBE_PROFILE" addons enable nvidia-device-plugin; tried=1; }
    sleep 5
  done
  [[ ${n:-0} -ge 1 ]] || die "no nvidia.com/gpu on the node (see the minikube NVIDIA tutorial, 'Troubleshooting')"
  ok "nvidia.com/gpu: $n"
elif [[ $GPU_MODE == dra ]]; then
  warn "DRA mode is experimental: the minikube docs and the NVIDIA DRA driver notes do not say whether the device plugin and the DRA driver can share a GPU."
  minikube -p "$MINIKUBE_PROFILE" addons disable nvidia-device-plugin || true
  log "Installing the NVIDIA DRA driver $DRA_DRIVER_VERSION"
  helm upgrade --install dra-driver-nvidia-gpu oci://registry.k8s.io/dra-driver-nvidia/charts/dra-driver-nvidia-gpu \
    --version "$DRA_DRIVER_VERSION" -n dra-driver-nvidia-gpu --create-namespace \
    --set gpuResourcesEnabledOverride=true --wait
fi

log "Installing the Kubeflow Spark Operator $SPARK_OPERATOR_VERSION (watching $NAMESPACE and $E2E_NAMESPACE)"
helm repo add spark-operator "$SPARK_OPERATOR_REPO" >/dev/null 2>&1 || true
helm repo update spark-operator >/dev/null
# The datalab chart owns the Spark ServiceAccount/Role, so the operator's per-namespace copies are switched off.
helm upgrade --install spark-operator spark-operator/spark-operator --version "$SPARK_OPERATOR_VERSION" \
  -n spark-operator --create-namespace \
  --set "spark.jobNamespaces={$NAMESPACE,$E2E_NAMESPACE}" \
  --set spark.serviceAccount.create=false --set spark.rbac.create=false \
  --wait

log "Installing Envoy Gateway $ENVOY_GATEWAY_VERSION (also installs the Gateway API CRDs)"
helm upgrade --install eg oci://docker.io/envoyproxy/gateway-helm --version "$ENVOY_GATEWAY_VERSION" \
  -n envoy-gateway-system --create-namespace --wait

if [[ $HEADLAMP == true ]]; then
  "$HELM_DIR/scripts/install-headlamp.sh"
fi

ok "Cluster ready. Next: helm/scripts/build-images.sh && helm/scripts/deploy.sh"
echo "Reach the Gateway from the host: helm/scripts/gateway-forward.sh   (or \`minikube tunnel\` + helm/scripts/hosts.sh)"
[[ $HEADLAMP == true ]] && echo "Headlamp login token: helm/scripts/headlamp-token.sh"
