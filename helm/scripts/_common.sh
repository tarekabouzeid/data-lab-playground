# shellcheck shell=bash disable=SC2034
# Sourced by the other helm/scripts/*.sh
set -euo pipefail

HELM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(cd "$HELM_DIR/.." && pwd)"
CHART="$HELM_DIR/datalab"
# shellcheck source=versions.env
source "$HELM_DIR/scripts/versions.env"

NAMESPACE="${NAMESPACE:-datalab}"
RELEASE="${RELEASE:-datalab}"
MINIKUBE_PROFILE="${MINIKUBE_PROFILE:-minikube}"

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m ok\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarn\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31mfail\033[0m %s\n' "$*" >&2; exit 1; }

require() {   # require <cmd> [hint]
  command -v "$1" >/dev/null 2>&1 || die "'$1' is required${2:+ ($2)}"
}

# API versions the chart's capability checks look for; needed by `helm template` (no cluster to ask).
HELM_API_VERSIONS=(
  --api-versions sparkoperator.k8s.io/v1alpha1/SparkConnect
  --api-versions sparkoperator.k8s.io/v1beta2/SparkApplication
  --api-versions gateway.networking.k8s.io/v1/Gateway
  --api-versions resource.k8s.io/v1/ResourceClaimTemplate
)
