#!/usr/bin/env bash
# Print a login token for Headlamp. Headlamp (in-cluster mode) signs users in with a ServiceAccount bearer token.
# This creates ServiceAccount `headlamp-login` bound to cluster-admin: fine for a single-user local minikube, do NOT
# do this on a shared cluster. Usage: helm/scripts/headlamp-token.sh [duration, default 8h]
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require kubectl
DURATION="${1:-8h}"
kubectl -n "$NAMESPACE" create serviceaccount headlamp-login --dry-run=client -o yaml | kubectl apply -f - >/dev/null
kubectl create clusterrolebinding headlamp-login-"$NAMESPACE" --clusterrole=cluster-admin \
  --serviceaccount="$NAMESPACE:headlamp-login" --dry-run=client -o yaml | kubectl apply -f - >/dev/null
warn "headlamp-login has cluster-admin: local single-user clusters only" 
kubectl -n "$NAMESPACE" create token headlamp-login --duration "$DURATION"
