#!/usr/bin/env bash
# Make the Envoy Gateway reachable from the host WITHOUT `minikube tunnel`, sudo or /etc/hosts edits:
# port-forward the Envoy proxy Service to localhost and use the *.localhost hostnames (gateway.domain default).
#   http://jupyter.datalab.localhost:8080/   http://headlamp.datalab.localhost:8080/   http://trino.datalab.localhost:8080/ ...
# Usage: helm/scripts/gateway-forward.sh [--port 8080]       Ctrl-C stops it.
# Alternatives: `minikube tunnel` + helm/scripts/hosts.sh (LoadBalancer address, standard port 80).
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require kubectl
PORT=8080
while [[ $# -gt 0 ]]; do
  case $1 in --port) PORT=$2; shift 2;; -h|--help) sed -n '2,7p' "$0"; exit 0;; *) die "unknown option: $1";; esac
done

# Envoy Gateway labels the proxy Service it creates for a Gateway (default namespace: envoy-gateway-system)
svc=$(kubectl get svc -A -l "gateway.envoyproxy.io/owning-gateway-name=datalab,gateway.envoyproxy.io/owning-gateway-namespace=$NAMESPACE" \
  -o jsonpath='{.items[0].metadata.namespace}/{.items[0].metadata.name}' 2>/dev/null || true)
[[ $svc == */* ]] || die "no Envoy proxy Service for Gateway $NAMESPACE/datalab yet: is the chart deployed with gateway.enabled=true, and Envoy Gateway running?"
ns=${svc%%/*}; name=${svc##*/}

hosts=$(kubectl -n "$NAMESPACE" get httproute -o jsonpath='{range .items[*]}{.spec.hostnames[*]}{" "}{end}' 2>/dev/null || true)
log "Forwarding localhost:$PORT -> svc/$name ($ns) :80"
for h in $hosts; do echo "  http://$h:$PORT/"; done
if [[ $hosts != *.localhost* ]]; then
  warn "gateway.domain is not *.localhost: add '127.0.0.1 $hosts' to /etc/hosts (see helm/scripts/hosts.sh --forward)"
fi
exec kubectl -n "$ns" port-forward "svc/$name" "$PORT:80"
