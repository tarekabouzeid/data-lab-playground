#!/usr/bin/env bash
# Build the lab's own images INTO minikube with the tags charts/data-lab expects
# (ghcr.io/tarekabouzeid/data-lab-playground/<name>:<appVersion>, pull policy IfNotPresent), and pre-pull the
# third-party images of the workstation profile so the first deploy does not stall.
#
# Usage: hack/workstation/build-images.sh [--force] [--load] [--no-pull]
#   (default) build inside minikube's Docker daemon: eval $(minikube docker-env)   [needs --container-runtime docker]
#   --load    build with the host Docker, then `minikube image load` (works with any runtime)
#   --force   rebuild even when the sources are older than the image
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

FORCE=false; LOAD=false; PULL=true
for a in "$@"; do
  case $a in
    --force) FORCE=true;; --load) LOAD=true;; --no-pull) PULL=false;;
    -h|--help) sed -n '2,10p' "$0"; exit 0;;
    *) die "unknown option: $a";;
  esac
done
require docker; require minikube; require helm

REGISTRY=ghcr.io/tarekabouzeid/data-lab-playground
SERVICES=(hive-metastore spark jupyter airflow-mcp)

if [[ $LOAD == false ]]; then
  eval "$(minikube -p "$MINIKUBE_PROFILE" docker-env)"
  log "Building inside minikube's Docker daemon"
else
  log "Building with the host Docker, then loading into minikube"
fi

needs_build() {   # image exists and is newer than every source file -> no build needed
  local svc=$1 img=$2
  [[ $FORCE == true ]] && return 0
  docker image inspect "$img" >/dev/null 2>&1 || return 0
  local created newest
  created=$(date -d "$(docker image inspect "$img" --format '{{.Created}}')" +%s 2>/dev/null || echo 0)
  newest=$(find "$ROOT/$svc" -type f -printf '%T@\n' | sort -nr | head -1 | cut -d. -f1)
  [[ -n $newest && $newest -gt $created ]]
}

for svc in "${SERVICES[@]}"; do
  img="$REGISTRY/$svc:$APP_VERSION"
  if needs_build "$svc" "$img"; then
    log "docker build $img"
    docker build -t "$img" "$ROOT/$svc"
  else
    ok "$img is up to date"
  fi
  if [[ $LOAD == true ]]; then minikube -p "$MINIKUBE_PROFILE" image load "$img"; fi
done

if [[ $PULL == true ]]; then
  # every other image the workstation profile renders (third-party, pinned in values.yaml)
  mapfile -t IMAGES < <(helm template "$RELEASE" "$CHART" -n "$NAMESPACE" -f "$PROFILE" "${HELM_API_VERSIONS[@]}" \
    | sed -n 's/^ *image: *"\{0,1\}\([^" ]*\)"\{0,1\}$/\1/p' | grep -v "^$REGISTRY/" | sort -u)
  for img in "${IMAGES[@]}"; do
    log "pull $img"
    if [[ $LOAD == true ]]; then
      docker pull "$img" && minikube -p "$MINIKUBE_PROFILE" image load "$img"
    else
      docker pull "$img"
    fi
  done
fi
ok "Images ready."
