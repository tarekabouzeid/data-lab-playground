#!/usr/bin/env python3
"""Static policy checks on `helm template` output of charts/data-lab.

Usage: helm template <release> charts/data-lab -n <ns> [...] | hack/check-chart-render.py <release-fullname> <ns>

Fails (exit 1) when a rendered object breaks a chart requirement:
  - cluster-scoped kind, or a namespace other than <ns>
  - a name not prefixed with <release-fullname>
  - missing app.kubernetes.io/{name,instance,version,managed-by,component} labels (objects and pod templates)
  - a Service that is not ClusterIP
  - a pod or container that misses the Pod Security Standard "restricted" settings
"""
import sys

import yaml

NAMESPACED = {"ConfigMap", "Secret", "Service", "Deployment", "PersistentVolumeClaim", "ServiceAccount",
              "Role", "RoleBinding", "NetworkPolicy", "Pod",
              "Gateway", "HTTPRoute",  # namespaced Gateway API kinds (gateway.enabled)
              "ResourceClaimTemplate"}  # DRA GPU claim (ollama.gpu.dra.enabled)
LABELS = ["app.kubernetes.io/name", "app.kubernetes.io/instance", "app.kubernetes.io/version",
          "app.kubernetes.io/managed-by", "helm.sh/chart"]


def pod_errors(where, spec, labels):
    errs = []
    for key in ("app.kubernetes.io/name", "app.kubernetes.io/instance", "app.kubernetes.io/component"):
        if key not in (labels or {}):
            errs.append(f"{where}: pod label {key} missing")
    psc = spec.get("securityContext") or {}
    if psc.get("runAsNonRoot") is not True:
        errs.append(f"{where}: pod runAsNonRoot is not true")
    if (psc.get("seccompProfile") or {}).get("type") not in ("RuntimeDefault", "Localhost"):
        errs.append(f"{where}: pod seccompProfile is not RuntimeDefault")
    if psc.get("runAsUser") == 0:
        errs.append(f"{where}: pod runs as uid 0")
    for f in ("hostNetwork", "hostPID", "hostIPC"):
        if spec.get(f):
            errs.append(f"{where}: {f} set")
    for vol in spec.get("volumes") or []:
        if "hostPath" in vol:
            errs.append(f"{where}: hostPath volume {vol['name']}")
    for c in (spec.get("initContainers") or []) + (spec.get("containers") or []):
        sc = c.get("securityContext") or {}
        cw = f"{where}/{c['name']}"
        if sc.get("allowPrivilegeEscalation") is not False:
            errs.append(f"{cw}: allowPrivilegeEscalation is not false")
        if sc.get("privileged"):
            errs.append(f"{cw}: privileged")
        if "ALL" not in ((sc.get("capabilities") or {}).get("drop") or []):
            errs.append(f"{cw}: capabilities.drop lacks ALL")
        if (sc.get("capabilities") or {}).get("add"):
            errs.append(f"{cw}: adds capabilities")
        if sc.get("runAsUser") == 0 or sc.get("runAsNonRoot") is False:
            errs.append(f"{cw}: runs as root")
        for p in c.get("ports") or []:
            if p.get("hostPort"):
                errs.append(f"{cw}: hostPort")
    return errs


def main():
    fullname, ns = sys.argv[1], sys.argv[2]
    errs, n = [], 0
    for doc in yaml.safe_load_all(sys.stdin):
        if not doc:
            continue
        n += 1
        kind, meta = doc["kind"], doc["metadata"]
        where = f"{kind}/{meta['name']}"
        if kind not in NAMESPACED:
            errs.append(f"{where}: kind is not in the namespaced allow-list")
        if meta.get("namespace") != ns:
            errs.append(f"{where}: namespace {meta.get('namespace')!r} != {ns!r}")
        if not meta["name"].startswith(fullname):
            errs.append(f"{where}: name not prefixed with {fullname}")
        for key in LABELS:
            if key not in (meta.get("labels") or {}):
                errs.append(f"{where}: label {key} missing")
        if kind == "Service" and doc["spec"].get("type", "ClusterIP") != "ClusterIP":
            errs.append(f"{where}: Service type {doc['spec']['type']}")
        if kind == "Deployment":
            tpl = doc["spec"]["template"]
            errs += pod_errors(where, tpl["spec"], tpl["metadata"].get("labels"))
        if kind == "Pod":
            errs += pod_errors(where, doc["spec"], meta.get("labels"))
        if kind == "ConfigMap" and "executor-pod-template.yaml" in doc.get("data", {}):
            tpl = yaml.safe_load(doc["data"]["executor-pod-template.yaml"])
            errs += pod_errors(f"{where}[executor]", tpl["spec"], tpl["metadata"].get("labels"))
    for e in errs:
        print("FAIL", e)
    print(f"checked {n} objects, {len(errs)} problems")
    sys.exit(1 if errs or n == 0 else 0)


if __name__ == "__main__":
    main()
