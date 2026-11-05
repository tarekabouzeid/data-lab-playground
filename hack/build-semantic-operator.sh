#!/usr/bin/env bash
# Build the Kubeflow Semantic Operator manager and server images from upstream source at the commit the chart was
# ported from (config layout in charts/data-lab/templates/_semantic.tpl, CRD in hack/crds/). The public
# public.ecr.aws/.../semantic-operator:v0.1.1 images predate that layout (env-based config, no --config flag).
#
# Usage: hack/build-semantic-operator.sh [tag]        (default: the chart's appVersion)
# Images: ghcr.io/tarekabouzeid/data-lab-playground/semantic-operator-{manager,server}:<tag>
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMMIT=4fd7057                                   # keep equal to the header of hack/crds/semantic.ossie.io_semanticmodels.yaml
TAG="${1:-$(sed -n 's/^appVersion: *"\{0,1\}\([^"]*\)"\{0,1\}$/\1/p' "$ROOT/charts/data-lab/Chart.yaml")}"
REPO=ghcr.io/tarekabouzeid/data-lab-playground
src=$(mktemp -d); trap 'rm -rf "$src"' EXIT
git clone -q https://github.com/kubeflow/semantic-operator "$src"
git -C "$src" checkout -q "$COMMIT"
for target in manager server; do
  echo "==> $REPO/semantic-operator-$target:$TAG (kubeflow/semantic-operator@$COMMIT)"
  docker build -q --target "$target" -t "$REPO/semantic-operator-$target:$TAG" \
    --label "org.opencontainers.image.source=https://github.com/tarekabouzeid/data-lab-playground" \
    --label "org.opencontainers.image.revision=kubeflow/semantic-operator@$COMMIT" "$src" >/dev/null
done
