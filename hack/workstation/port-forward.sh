#!/usr/bin/env bash
# kubectl port-forward for the ports that are NOT routed through the Gateway (HMS, Spark Connect gRPC, OTLP) and,
# with --all, the HTTP UIs too, on the localhost ports docker compose publishes. Ctrl-C stops everything.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require kubectl
ALL=false; [[ ${1:-} == --all ]] && ALL=true
pids=(); trap 'kill "${pids[@]}" 2>/dev/null' EXIT INT TERM
fwd() { kubectl -n "$NAMESPACE" port-forward "svc/$FULLNAME-$1" "${@:2}" >/dev/null & pids+=($!); echo "  localhost:${2%%:*} -> $FULLNAME-$1 :${2##*:}"; }
log "Forwarding (namespace $NAMESPACE, release $RELEASE)"
fwd hms 9083:9083 9084:9084
fwd spark-connect 15002:15002
fwd qdrant 6334:6334
fwd phoenix 4317:4317
if $ALL; then
  fwd jupyter 8888:8888; fwd trino 8080:8080; fwd phoenix 6006:6006; fwd ollama 11434:11434
  fwd qdrant 6333:6333; fwd seaweedfs 8333:8333; fwd spark-connect 4040:4040; fwd semantic 8090:8090
fi
echo "Spark Connect needs the token: sc://localhost:15002/;authorization=Bearer \$(kubectl -n $NAMESPACE get secret $FULLNAME-spark-connect -o jsonpath='{.data.token}' | base64 -d)"
wait
