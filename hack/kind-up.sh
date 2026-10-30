#!/usr/bin/env bash
# Local setup for hack/chart-e2e-kind.sh: create the kind cluster and load the chart's images into it
# (CI does the same in .github/workflows/chart-e2e.yaml).
#
#   --calico    replace kindnet with Calico. kindnet enforces NetworkPolicy through NFQUEUE, which some kernels
#               lack (WSL2: CONFIG_NFT_QUEUE is not set); there the `parallel` scenario only warns that the CNI
#               does not enforce policies. Calico's iptables mode needs no NFQUEUE.
#   --no-build  use the ghcr.io/tarekabouzeid/data-lab-playground/*:<appVersion> images already in Docker
#
# Images are loaded with `docker save --platform linux/amd64` + `kind load image-archive`: `kind load docker-image`
# fails on multi-arch images from Docker's containerd image store ("content digest ... not found").
#
# Usage: hack/kind-up.sh [--calico] [--no-build] [--name datalab]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CALICO_VERSION=v3.32.2
NAME=datalab; CALICO=false; BUILD=true
while [[ $# -gt 0 ]]; do
  case $1 in
    --calico) CALICO=true; shift;; --no-build) BUILD=false; shift;; --name) NAME=$2; shift 2;;
    -h|--help) sed -n '2,14p' "$0"; exit 0;;
    *) echo "unknown option: $1" >&2; exit 1;;
  esac
done

APP=$(awk '/^appVersion:/ {gsub(/"/, "", $2); print $2}' "$ROOT/charts/data-lab/Chart.yaml")
REPO=ghcr.io/tarekabouzeid/data-lab-playground
OWN=("$REPO/hive-metastore:$APP" "$REPO/spark:$APP" "$REPO/airflow-mcp:$APP"
     "$REPO/semantic-operator-manager:$APP" "$REPO/semantic-operator-server:$APP")
# keep in sync with the "Load images into kind" step of .github/workflows/chart-e2e.yaml
PULLED=(chrislusf/seaweedfs:4.48 trinodb/trino:483 postgres:17.11 apache/airflow:3.3.2-python3.12 busybox:1.36
        ghcr.io/kubeflow/mcp-apache-spark-history-server:1.0.1
)

if kind get clusters 2>/dev/null | grep -qx "$NAME"; then
  echo "==> kind cluster $NAME exists, reusing it"
elif [[ $CALICO == true ]]; then
  echo "==> Creating kind cluster $NAME without kindnet, then Calico $CALICO_VERSION"
  kind create cluster --name "$NAME" --config - <<'EOF'
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
networking:
  disableDefaultCNI: true
  podSubnet: 192.168.0.0/16
EOF
  kubectl apply -f "https://raw.githubusercontent.com/projectcalico/calico/$CALICO_VERSION/manifests/calico.yaml" >/dev/null
  # WSL2 has no securityfs: the sys-kernel-security hostPath then fails with "mkdir /sys/kernel/security: read-only
  # file system" (CreateContainerError), and no mount target can be created there. Calico only reads it to detect the
  # LSM, so drop the volume and its mount.
  if ! docker exec "$NAME-control-plane" test -d /sys/kernel/security; then
    patch=$(kubectl -n kube-system get ds calico-node -o json | python3 -c '
import json, sys
spec = json.load(sys.stdin)["spec"]["template"]["spec"]
ops = [{"op": "remove", "path": f"/spec/template/spec/volumes/{i}"}
       for i, v in enumerate(spec["volumes"]) if v["name"] == "sys-kernel-security"]
for c, ctr in enumerate(spec["containers"]):
    ops += [{"op": "remove", "path": f"/spec/template/spec/containers/{c}/volumeMounts/{i}"}
            for i, m in enumerate(ctr.get("volumeMounts", [])) if m["name"] == "sys-kernel-security"]
print(json.dumps(ops))')
    kubectl -n kube-system patch ds calico-node --type=json -p "$patch" >/dev/null
  fi
  kubectl -n kube-system rollout status ds/calico-node --timeout=300s
  kubectl wait --for=condition=Ready node --all --timeout=300s
else
  echo "==> Creating kind cluster $NAME"
  kind create cluster --name "$NAME"
fi
kubectl config use-context "kind-$NAME" >/dev/null

if [[ $BUILD == true ]]; then
  for c in hive-metastore spark airflow-mcp; do
    echo "==> Building $REPO/$c:$APP"
    docker build -q -t "$REPO/$c:$APP" "$ROOT/$c" >/dev/null
  done
  "$ROOT/hack/build-semantic-operator.sh" "$APP"   # semantic scenario
fi
for i in "${PULLED[@]}"; do docker image inspect "$i" >/dev/null 2>&1 || docker pull -q "$i" >/dev/null; done

tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
for i in "${OWN[@]}" "${PULLED[@]}"; do
  echo "==> Loading $i"
  docker save --platform linux/amd64 -o "$tmp/img.tar" "$i"
  kind load image-archive --name "$NAME" "$tmp/img.tar" >/dev/null
done
rm -f "$tmp/img.tar"
echo "==> Ready: hack/chart-e2e-kind.sh [default|minimal|parallel|semantic|all]   (kind delete cluster --name $NAME when done)"
