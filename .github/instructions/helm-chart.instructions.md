---
description: "Use when editing the Helm chart under charts/data-lab (templates, values, schema, ci profiles) or its CI workflows."
applyTo: "charts/**"
---

# Helm chart (`charts/data-lab`)

- `values.yaml` is a **public API**: consumers (Sandpiper) set keys programmatically. Add keys, do not rename or remove them
  (that needs a major chart version). Every key gets a helm-docs `# --` comment.
- After editing `values.yaml`: run `hack/gen-chart-schema.py` (strict, closed schema) and `helm-docs -c charts -s file`.
  CI fails if `values.schema.json` or `README.md` drift.
- Every chart change bumps `Chart.yaml` `version` and adds a `CHANGELOG.md` entry. `appVersion` is the lab release, and
  the `ghcr.io/.../data-lab-playground/*` image tags in values must equal it.
- Names: always `include "datalab.componentName" (list $ "<component>")`. Hosts: `datalab.host`. Never a literal name,
  namespace, or `default` namespace. Resources must be namespaced (no ClusterRole, CRD, PV, StorageClass, etc.).
- Every pod uses `datalab.podSpecCommon` + `datalab.containerSecurityContext` (PSS restricted), including init containers,
  test pods and the Spark executor pod template. Services are ClusterIP.
- Credentials: `existingSecret` + key name in values, else a generated Secret in `templates/secrets.yaml` via
  `datalab.secretValue` (lookup keeps it stable across upgrades). List each one in `endpoints-configmap.yaml`.
- New component: toggle `<component>.enabled`; add render-time checks to `templates/validate.yaml` for its dependencies;
  add it to the endpoints ConfigMap, the README component/credential/footprint tables and `docs/VERSIONS.md` §8.
- Optional features that need an API (Gateway API, DRA, the SemanticModel CRD) check `.Capabilities` unless
  `global.apiChecks=false` (offline `helm lint` only). New namespaced kinds go into `hack/check-chart-render.py`'s allow-list.
- Defaults are frozen for Sandpiper: `tests/chart/golden/default-resources.txt` must not change unless the default render
  changes on purpose (say so in the CHANGELOG). Write a render test in `tests/chart/` for every new value (test first).
- Cluster-scoped prerequisites (CRDs, GatewayClass, PVs, operators) belong in `hack/workstation/` or `hack/crds/`, never in
  the chart.
- Check before pushing:
  `python3 -m pytest tests/chart tests/unit -q && hack/check-compose-parity.py`
  `for f in charts/data-lab/ci/*-values.yaml; do helm lint --strict charts/data-lab --kube-version 1.34.0 -f $f --set global.apiChecks=false; helm template r charts/data-lab -n ns -f $f --api-versions gateway.networking.k8s.io/v1/Gateway --api-versions semantic.ossie.io/v1alpha1/SemanticModel --api-versions resource.k8s.io/v1/ResourceClaimTemplate | hack/check-chart-render.py r-data-lab ns; done`
