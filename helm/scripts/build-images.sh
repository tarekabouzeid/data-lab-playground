#!/usr/bin/env bash
# Build the four local images (same Dockerfiles and tags as start-platform.sh) INTO minikube, and pre-pull the pinned
# third-party images so the first deploy does not stall. Pull policy in the chart is IfNotPresent.
#
# Usage: helm/scripts/build-images.sh [--force] [--load] [--no-pull]
#   (default) build inside minikube's Docker daemon: eval $(minikube docker-env)   [needs --container-runtime docker]
#   --load    build with the host Docker, then `minikube image load` (works with any runtime)
#   --force   rebuild even when the sources are older than the image
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

FORCE=false; LOAD=false; PULL=true
for a in "$@"; do
  case $a in
    --force) FORCE=true;; --load) LOAD=true;; --no-pull) PULL=false;;
    -h|--help) sed -n '2,8p' "$0"; exit 0;;
    *) die "unknown option: $a";;
  esac
done
require docker; require minikube

SERVICES=(hive-metastore trino spark jupyter)

if [[ $LOAD == false ]]; then
  eval "$(minikube -p "$MINIKUBE_PROFILE" docker-env)"
  log "Building inside minikube's Docker daemon"
else
  log "Building with the host Docker, then loading into minikube"
fi

needs_build() {   # image exists and is newer than every source file -> no build needed
  local svc=$1 img="datalab-playground/$1"
  [[ $FORCE == true ]] && return 0
  docker image inspect "$img" >/dev/null 2>&1 || return 0
  local created newest
  created=$(date -d "$(docker image inspect "$img" --format '{{.Created}}')" +%s 2>/dev/null || echo 0)
  newest=$(find "$ROOT/$svc" -type f -printf '%T@\n' | sort -nr | head -1 | cut -d. -f1)
  [[ -n $newest && $newest -gt $created ]]
}

for svc in "${SERVICES[@]}"; do
  if needs_build "$svc"; then
    log "docker build datalab-playground/$svc"
    docker build -t "datalab-playground/$svc" "$ROOT/$svc"
  else
    ok "datalab-playground/$svc is up to date"
  fi
  if [[ $LOAD == true ]]; then minikube -p "$MINIKUBE_PROFILE" image load "datalab-playground/$svc:latest"; fi
done

if [[ $PULL == true ]]; then
  # third-party images come from values.yaml (the same list check-parity.sh compares with docker-compose.yaml)
  mapfile -t IMAGES < <(python3 - "$CHART/values.yaml" <<'PY'
import re, sys
for repo, tag in re.findall(r'repository:\s*([^\s,}]+),\s*tag:\s*"?([^"\s}]+)"?', open(sys.argv[1]).read()):
    if not repo.startswith("datalab-playground/"):
        print(f"{repo}:{tag}")
PY
)
  for img in "${IMAGES[@]}"; do
    log "pull $img"
    if [[ $LOAD == true ]]; then
      docker pull "$img" && minikube -p "$MINIKUBE_PROFILE" image load "$img"
    else
      docker pull "$img"
    fi
  done
fi
ok "Images ready:"; minikube -p "$MINIKUBE_PROFILE" image ls | grep -E 'datalab-playground|seaweedfs|postgres|phoenix|ollama|qdrant' || true
