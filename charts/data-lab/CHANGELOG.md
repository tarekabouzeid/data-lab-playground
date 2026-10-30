# data-lab chart changelog

Chart versions follow semver. `appVersion` is the lab release the chart deploys (see
[RELEASE_NOTES.md](../../RELEASE_NOTES.md) and [docs/VERSIONS.md](../../docs/VERSIONS.md)).

## 0.2.0 — unreleased (appVersion 3.0.0)

Merges the former `helm/datalab` minikube chart into this one and adds the Kubeflow Semantic Operator. Additive only:
with default values the chart renders the same resources as 0.1.0 (`tests/chart/golden/default-resources.txt`).

### Added
- `semantic` (off): Kubeflow Semantic Operator manager + server (v0.1.1) on the release's Trino, serving an Apache Ossie
  semantic layer over MCP (`/mcp`) and REST (`/v1`); `semantic.models.dbtDemo` SemanticModel over the dbt demo tables
  (certified metrics, roles `analyst` / `marketing`); `semantic.models.extra`; a pre-delete hook that removes the
  release's SemanticModels before the manager goes (its finalizer would block uninstall); endpoints keys
  `SEMANTIC_MCP_URL`, `SEMANTIC_REST_URL`, `SEMANTIC_DEFAULT_ROLE`, `SEMANTIC_DEMO_MODEL`. Needs the SemanticModel CRD
  (`hack/crds/`, installed separately). One semantic-enabled release per namespace (refused at install).
- `gateway` (off): Gateway API HTTPRoutes for every enabled UI on a namespaced Gateway (`gatewayClassName`) or an existing
  one (`parentRef`), hostnames `{{route}}.{{release}}.{{domain}}`, `extraRoutes`, `allowFromNamespaces` for the
  NetworkPolicy, Trino forwarded headers, `GATEWAY_URL_<ROUTE>` endpoints keys. No GatewayClass is rendered.
- `ollama.gpu.dra`: claim the GPU with a namespaced ResourceClaimTemplate (DRA) instead of `nvidia.com/gpu`.
- `spark.applications.enabled`: ServiceAccount + Role for SparkApplication drivers (Kubeflow Spark Operator).
- `tests.lakehouseE2e.enabled`: the 19-check lakehouse e2e as a second `helm test` pod (Spark Connect + Trino).
- `jupyter.persistence.existingClaim`; Jupyter now also gets `TRINO_HOST`/`TRINO_PORT` and the S3 keys (from the
  S3 Secret), so every notebook runs unchanged on the chart.
- `global.apiChecks` (default true): turn the Gateway/DRA/SemanticModel API checks off for offline `helm lint` only.
- Profiles `ci/semantic-values.yaml` and `ci/workstation-values.yaml` (minikube + GPU lab, `hack/workstation/`).
- CI: render tests (`tests/chart`), unit tests, compose parity check, kind e2e scenario `semantic`.

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
- Airflow DAGs from a PVC (existing or chart-made) or git-sync. Task logs on a PVC by default
  (`airflow.logs.persistence.enabled: true`), so finished runs' logs survive pod restarts.
- `helm test`: Spark writes an Iceberg table through Spark Connect, Trino reads it, then it is dropped.
- Spark History Server (`sparkHistory`, on): reads Spark event logs from `s3a://<bucket>/spark-events`; Spark Connect
  writes rolled event logs there and now shuts down gracefully (SIGTERM reaches the JVM), closing its log.
- MCP servers for agents (off by default): Kubeflow Spark History MCP 1.0.1 (`sparkHistoryMcp`, servers `history`
  and `live`) and Astronomer Airflow MCP 0.9.1 (`airflowMcp`, own Airflow user with `viewer` role when read-only).
- CI: helm lint, ct lint, kubeconform, schema/README drift checks, kind e2e (default, minimal, two parallel
  releases in one namespace and across namespaces, helm test, clean uninstall), OCI publish on `v*` tags.
