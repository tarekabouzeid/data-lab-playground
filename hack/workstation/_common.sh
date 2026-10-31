# shellcheck shell=bash disable=SC2034
# Sourced by the other hack/workstation/*.sh: the workstation lab is charts/data-lab with ci/workstation-values.yaml,
# release `datalab` in namespace `datalab` (so the gateway hostnames are <route>.datalab.localhost).
set -euo pipefail

WS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$WS_DIR/../.." && pwd)"
CHART="$ROOT/charts/data-lab"
PROFILE="$CHART/ci/workstation-values.yaml"
# shellcheck source=versions.env
source "$WS_DIR/versions.env"

NAMESPACE="${NAMESPACE:-datalab}"
RELEASE="${RELEASE:-datalab}"
FULLNAME="$RELEASE-data-lab"
MINIKUBE_PROFILE="${MINIKUBE_PROFILE:-minikube}"
APP_VERSION="$(sed -n 's/^appVersion: *"\{0,1\}\([^"]*\)"\{0,1\}$/\1/p' "$CHART/Chart.yaml")"

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m ok\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarn\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31mfail\033[0m %s\n' "$*" >&2; exit 1; }

require() {   # require <cmd> [hint]
  command -v "$1" >/dev/null 2>&1 || die "'$1' is required${2:+ ($2)}"
}

# API versions the chart checks for (gateway, semantic, DRA); `helm template` has no cluster to ask.
HELM_API_VERSIONS=(
  --api-versions gateway.networking.k8s.io/v1/Gateway
  --api-versions semantic.ossie.io/v1alpha1/SemanticModel
  --api-versions resource.k8s.io/v1/ResourceClaimTemplate
)
