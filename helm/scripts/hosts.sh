#!/usr/bin/env bash
# Print the /etc/hosts line for the Gateway hostnames. It edits nothing: review it, then add it yourself.
#   helm/scripts/hosts.sh            LoadBalancer address (needs `minikube tunnel` running in another terminal), port 80
#   helm/scripts/hosts.sh --forward  127.0.0.1, for helm/scripts/gateway-forward.sh (only needed if *.localhost does not
#                                    resolve on your machine, or when gateway.domain is not *.localhost)
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require kubectl
hosts=$(kubectl -n "$NAMESPACE" get httproute -o jsonpath='{range .items[*]}{.spec.hostnames[*]}{" "}{end}')
[[ -n $hosts ]] || die "no HTTPRoutes in namespace $NAMESPACE: is the chart deployed with gateway.enabled=true?"
if [[ ${1:-} == --forward ]]; then
  addr=127.0.0.1; port=":8080"
else
  addr=$(kubectl -n "$NAMESPACE" get gateway datalab -o jsonpath='{.status.addresses[0].value}' 2>/dev/null || true)
  [[ -n $addr ]] || die "the Gateway has no address yet: is \`minikube tunnel\` running? (or use --forward with gateway-forward.sh)"
  port=""
fi
echo "# add to /etc/hosts:"
echo "$addr $hosts"
echo
for h in $hosts; do echo "  http://$h$port/"; done
