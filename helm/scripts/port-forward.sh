#!/usr/bin/env bash
# kubectl port-forward for the ports that are NOT routed through the Gateway (databases, HMS, gRPC) and, with --all,
# the HTTP UIs too, on the same localhost ports docker compose publishes. Ctrl-C stops everything.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require kubectl
ALL=false; [[ ${1:-} == --all ]] && ALL=true
pids=(); trap 'kill "${pids[@]}" 2>/dev/null' EXIT INT TERM
fwd() { kubectl -n "$NAMESPACE" port-forward "$1" "${@:2}" >/dev/null & pids+=($!); echo "  localhost:${2%%:*} -> $1 :${2##*:}"; }
log "Forwarding (namespace $NAMESPACE)"
fwd svc/metastore-db 5433:5432
fwd svc/db 5432:5432
fwd svc/hive-metastore 9083:9083 9084:9084
fwd svc/datalab-spark-server 15002:15002
fwd svc/qdrant 6334:6334
fwd svc/phoenix 4317:4317
if $ALL; then
  fwd svc/jupyter 8888:8888; fwd svc/trino 8080:8080; fwd svc/phoenix 6006:6006; fwd svc/ollama 11434:11434
  fwd svc/qdrant 6333:6333; fwd svc/seaweedfs-http 8333:8333; fwd svc/datalab-spark-server 4040:4040
fi
wait
