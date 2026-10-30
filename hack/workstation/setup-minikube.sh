#!/usr/bin/env bash
# Create a local minikube cluster (optionally with an NVIDIA GPU) and install everything cluster-wide the workstation
# lab needs, so charts/data-lab itself stays namespaced: Kubeflow Spark Operator, Envoy Gateway + GatewayClass
# `datalab`, the SemanticModel CRD, the notebooks PersistentVolume and, optionally, the NVIDIA DRA driver.
# GPU steps follow https://minikube.sigs.k8s.io/docs/tutorials/nvidia/ (docker driver, Linux only).
#
# Usage: hack/workstation/setup-minikube.sh [--cpus 8] [--memory 24g (default: 24g or 75% of Docker's memory, whichever is less)] [--disk-size 80g]
#                                       [--gpu-mode device-plugin|dra|none] [--profile minikube] [--no-headlamp]
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

CPUS=8; MEMORY=""; DISK=80g; GPU_MODE=device-plugin; HEADLAMP=true
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

# minikube refuses more memory than Docker has (WSL2 gives Docker only part of the host's RAM)
docker_mib=$(( $(docker info --format '{{.MemTotal}}') / 1048576 ))
if [[ -z $MEMORY ]]; then
  MEMORY="$(( docker_mib * 3 / 4 < 24576 ? docker_mib * 3 / 4 : 24576 ))m"
elif [[ $MEMORY =~ ^([0-9]+)([gGmM])b?$ ]]; then
  want=${BASH_REMATCH[1]}; [[ ${BASH_REMATCH[2]} =~ [gG] ]] && want=$(( want * 1024 ))
  (( want <= docker_mib )) || warn "--memory $MEMORY is more than Docker's ${docker_mib} MiB; minikube will refuse it"
fi
log "Starting minikube $MINIKUBE_VERSION / Kubernetes $KUBERNETES_VERSION (profile $MINIKUBE_PROFILE)"
# --container-runtime docker: required by the GPU docs, and what makes `eval $(minikube docker-env)` builds work.
# The mount makes jupyter/notebooks editable from the host (notebooks-pv.yaml, chart: jupyter.persistence.existingClaim).
minikube start -p "$MINIKUBE_PROFILE" --driver docker --container-runtime docker \
  --kubernetes-version "$KUBERNETES_VERSION" --cpus "$CPUS" --memory "$MEMORY" --disk-size "$DISK" \
  ${GPU_ARGS[@]+"${GPU_ARGS[@]}"} \
  --mount --mount-string "$ROOT/jupyter/notebooks:/datalab/notebooks"
kubectl config use-context "$MINIKUBE_PROFILE" >/dev/null

# charts/data-lab is "restricted"-clean; Headlamp and the operator's driver pods share the namespace, so warn/audit only.
log "Creating namespace $NAMESPACE (Pod Security: warn/audit at restricted)"
kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f - >/dev/null
kubectl label namespace "$NAMESPACE" --overwrite \
  pod-security.kubernetes.io/warn=restricted pod-security.kubernetes.io/audit=restricted >/dev/null

log "Notebooks volume: $ROOT/jupyter/notebooks -> PV/PVC datalab-notebooks"
sed "s/NAMESPACE/$NAMESPACE/g" "$WS_DIR/notebooks-pv.yaml" | kubectl apply -f - >/dev/null

log "SemanticModel CRD (Kubeflow Semantic Operator, cluster-scoped)"
kubectl apply -f "$ROOT/hack/crds/semantic.ossie.io_semanticmodels.yaml" >/dev/null

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

log "Installing the Kubeflow Spark Operator $SPARK_OPERATOR_VERSION (watching $NAMESPACE)"
helm repo add spark-operator "$SPARK_OPERATOR_REPO" >/dev/null 2>&1 || true
helm repo update spark-operator >/dev/null
# charts/data-lab owns the SparkApplication ServiceAccount/Role (spark.applications), so the operator's copies are off.
helm upgrade --install spark-operator spark-operator/spark-operator --version "$SPARK_OPERATOR_VERSION" \
  -n spark-operator --create-namespace \
  --set "spark.jobNamespaces={$NAMESPACE}" \
  --set spark.serviceAccount.create=false --set spark.rbac.create=false \
  --wait

log "Installing Envoy Gateway $ENVOY_GATEWAY_VERSION (also installs the Gateway API CRDs)"
helm upgrade --install eg oci://docker.io/envoyproxy/gateway-helm --version "$ENVOY_GATEWAY_VERSION" \
  -n envoy-gateway-system --create-namespace --wait
log "GatewayClass datalab (cluster-scoped, so here and not in the chart)"
kubectl apply -f - >/dev/null <<YAML
apiVersion: gateway.networking.k8s.io/v1
kind: GatewayClass
metadata:
  name: datalab
spec:
  controllerName: gateway.envoyproxy.io/gatewayclass-controller
YAML

if [[ $HEADLAMP == true ]]; then
  "$WS_DIR/install-headlamp.sh"
fi

ok "Cluster ready. Next: hack/workstation/build-images.sh && hack/workstation/deploy.sh"
echo "Reach the Gateway from the host: hack/workstation/gateway-forward.sh"
if [[ $HEADLAMP == true ]]; then echo "Headlamp login token: hack/workstation/headlamp-token.sh"; fi
