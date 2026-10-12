# data-lab chart changelog

Chart versions follow semver. `appVersion` is the lab release the chart deploys (see
[RELEASE_NOTES.md](../../RELEASE_NOTES.md) and [docs/VERSIONS.md](../../docs/VERSIONS.md)).

## 0.1.0 — unreleased (appVersion 3.0.0)

First version of the chart.

### Added
- Components with `<component>.enabled` toggles: SeaweedFS, Hive Metastore 4.2.1 (Thrift + Iceberg REST) with
  Postgres, Spark Connect 4.1.3 (Iceberg 1.12.0, executors as pods), Trino 483, Airflow 3.3.2 (LocalExecutor) with
  Postgres. Optional and off by default: JupyterLab, Ollama, Qdrant, Phoenix.
- `<fullname>-endpoints` ConfigMap: every enabled endpoint plus the Secret name/key of its credential
  (flat keys + `ENDPOINTS_JSON`).
- Credentials: `existingSecret` for every credential, otherwise generated once and kept across upgrades.
- Restricted pod security everywhere (Spark executors via pod template). Namespaced RBAC only. ClusterIP
  Services only. NetworkPolicy limited to the same release (+ `networkPolicy.extraIngressFrom`).
- Render-time validation of component combinations. Strict `values.schema.json` (unknown keys rejected).
- Airflow DAGs from a PVC (existing or chart-made) or git-sync.
- `helm test`: Spark writes an Iceberg table through Spark Connect, Trino reads it, then it is dropped.
- CI: helm lint, ct lint, kubeconform, schema/README drift checks, kind e2e (default, minimal, two parallel
  releases in one namespace and across namespaces, helm test, clean uninstall), OCI publish on `v*` tags.
