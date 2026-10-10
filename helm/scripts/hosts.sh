#!/usr/bin/env bash
# Print the /etc/hosts line for the Gateway hostnames. It edits nothing: review it, then add it yourself.
# Needs `minikube tunnel` running in another terminal so the Envoy Gateway Service gets an address.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require kubectl
addr=$(kubectl -n "$NAMESPACE" get gateway datalab -o jsonpath='{.status.addresses[0].value}' 2>/dev/null || true)
[[ -n $addr ]] || die "the Gateway has no address yet: is \`minikube tunnel\` running, and is the chart deployed with gateway.enabled=true?"
hosts=$(kubectl -n "$NAMESPACE" get httproute -o jsonpath='{range .items[*]}{.spec.hostnames[*]}{" "}{end}')
echo "# add to /etc/hosts:"
echo "$addr $hosts"
echo
for h in $hosts; do echo "  http://$h/"; done
