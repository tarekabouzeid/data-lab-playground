# data-lab Helm chart

![Version: 0.1.0](https://img.shields.io/badge/Version-0.1.0-informational?style=flat-square) ![Type: application](https://img.shields.io/badge/Type-application-informational?style=flat-square) ![AppVersion: 3.0.0](https://img.shields.io/badge/AppVersion-3.0.0-informational?style=flat-square)

Ephemeral lakehouse lab: Hive Metastore (Thrift + Iceberg REST), SeaweedFS S3, Spark Connect, Trino and Airflow, with optional Jupyter and GenAI services (Ollama, Qdrant, Phoenix). Built for many parallel releases on one cluster.

The chart is built to be installed **programmatically, once per user request, many releases in parallel on
one cluster** (Sandpiper). `values.yaml` is a public API, `values.schema.json` rejects unknown keys, and the
`<release>-data-lab-endpoints` ConfigMap is the contract consumers read back.

Component versions are pinned in [docs/VERSIONS.md](../../docs/VERSIONS.md) (section 8).

## Install

```bash
# from the OCI registry (published on every v* tag)
helm install lab1 oci://ghcr.io/tarekabouzeid/charts/data-lab --version 0.1.0 \
  --namespace lab1 --create-namespace --wait --timeout 10m

# minimal lakehouse profile: hms + seaweedfs + spark + trino
helm install lab1 oci://ghcr.io/tarekabouzeid/charts/data-lab --version 0.1.0 \
  -n lab1 --create-namespace -f https://raw.githubusercontent.com/tarekabouzeid/data-lab-playground/main/charts/data-lab/ci/minimal-values.yaml

helm test lab1 -n lab1      # Spark writes an Iceberg table, Trino reads it, then it is dropped
helm uninstall lab1 -n lab1 # removes everything, PVCs and Secrets included
```

## Components

| Component | Toggle | Default | Image | Port(s) | Needs |
|---|---|---|---|---|---|
| SeaweedFS (S3) | `seaweedfs.enabled` | on | `chrislusf/seaweedfs:4.48` | 8333 | — |
| Hive Metastore (+ Postgres) | `hms.enabled` | on | `ghcr.io/tarekabouzeid/data-lab-playground/hive-metastore` (HMS 4.2.1), `postgres:17.11` | 9083 Thrift, 9084 Iceberg REST | S3 |
| Spark Connect | `spark.enabled` | on | `ghcr.io/tarekabouzeid/data-lab-playground/spark` (Spark 4.1.3, Iceberg 1.12.0) | 15002 gRPC, 4040 UI | hms |
| Trino | `trino.enabled` | on | `trinodb/trino:483` | 8080 | hms |
| Airflow (+ Postgres) | `airflow.enabled` | on | `apache/airflow:3.3.2-python3.12`, `postgres:17.11` | 8080 | — |
| Spark History Server | `sparkHistory.enabled` | on | same image as Spark | 18080 | spark |
| Spark History MCP (Kubeflow) | `sparkHistoryMcp.enabled` | off | `ghcr.io/kubeflow/mcp-apache-spark-history-server:1.0.1` | 18888 (`/mcp`) | sparkHistory |
| Airflow MCP (Astronomer) | `airflowMcp.enabled` | off | `ghcr.io/tarekabouzeid/data-lab-playground/airflow-mcp` (astro-airflow-mcp 0.9.1) | 8000 (`/mcp`) | airflow |
| JupyterLab | `jupyter.enabled` | off | `ghcr.io/tarekabouzeid/data-lab-playground/jupyter` | 8888 | — |
| Ollama | `ollama.enabled` | off | `ollama/ollama:0.40.3` | 11434 | GPU recommended |
| Qdrant | `qdrant.enabled` | off | `qdrant/qdrant:v1.19.2-unprivileged` | 6333 HTTP, 6334 gRPC | — |
| Phoenix | `phoenix.enabled` | off | `arizephoenix/phoenix:version-20.21.0-nonroot` | 6006 UI/OTLP-HTTP, 4317 OTLP-gRPC | — |

Invalid combinations fail at render with a message, for example `trino.enabled=true requires hms.enabled=true`.
Object storage can be external: `seaweedfs.enabled=false`, `s3.endpoint=https://...`, `s3.auth.existingSecret=...`.

The `ghcr.io/tarekabouzeid/data-lab-playground/*` images are built from this repo's Dockerfiles and pushed by
`.github/workflows/chart-release.yaml` with the lab release as tag (= `appVersion`).

## Consumer contract: `<fullname>-endpoints`

Every enabled endpoint, plus the name and key of the Secret holding its credential. Flat keys work with `envFrom`;
`ENDPOINTS_JSON` holds the same data structured.

| Key | Example (release `lab1`, namespace `lab1`) |
|---|---|
| `S3_ENDPOINT`, `S3_BUCKET`, `S3_REGION`, `S3_WAREHOUSE` | `http://lab1-data-lab-seaweedfs.lab1.svc:8333`, `warehouse`, `us-east-1`, `s3a://warehouse/` |
| `S3_CREDENTIALS_SECRET`, `S3_CREDENTIALS_ACCESS_KEY_KEY`, `S3_CREDENTIALS_SECRET_KEY_KEY` | `lab1-data-lab-s3`, `access-key`, `secret-key` |
| `HMS_THRIFT_URI`, `ICEBERG_REST_URI` | `thrift://lab1-data-lab-hms.lab1.svc:9083`, `http://lab1-data-lab-hms.lab1.svc:9084/iceberg` |
| `TRINO_URL`, `TRINO_CATALOGS` | `http://lab1-data-lab-trino.lab1.svc:8080`, `hive,iceberg,lakehouse` |
| `SPARK_CONNECT_URL`, `SPARK_CONNECT_AUTH`, `SPARK_CONNECT_CREDENTIALS_SECRET`, `SPARK_CONNECT_CREDENTIALS_TOKEN_KEY` | `sc://lab1-data-lab-spark-connect.lab1.svc:15002`, `bearer`, `lab1-data-lab-spark-connect`, `token` |
| `SPARK_UI_URL`, `SPARK_ICEBERG_CATALOG` | `http://lab1-data-lab-spark-connect.lab1.svc:4040`, `iceberg_catalog` |
| `SPARK_HISTORY_URL`, `SPARK_EVENT_LOG_DIR` | `http://lab1-data-lab-spark-history.lab1.svc:18080`, `s3a://warehouse/spark-events` |
| `SPARK_HISTORY_MCP_URL`, `AIRFLOW_MCP_URL` | `http://lab1-data-lab-spark-history-mcp.lab1.svc:18888/mcp`, `http://lab1-data-lab-airflow-mcp.lab1.svc:8000/mcp` (when enabled) |
| `AIRFLOW_URL`, `AIRFLOW_API_URL`, `AIRFLOW_USERNAME`, `AIRFLOW_CREDENTIALS_SECRET`, `AIRFLOW_CREDENTIALS_PASSWORD_KEY` | `http://lab1-data-lab-airflow.lab1.svc:8080`, `.../api/v2`, `admin`, `lab1-data-lab-airflow`, `password` |
| `AIRFLOW_DAGS_SOURCE`, `AIRFLOW_DAGS_PVC` | `pvc`, `lab1-data-lab-airflow-dags` |
| `JUPYTER_URL`, `JUPYTER_CREDENTIALS_SECRET`, `JUPYTER_CREDENTIALS_TOKEN_KEY` | when enabled |
| `OLLAMA_URL` | when enabled (no auth) |
| `QDRANT_URL`, `QDRANT_GRPC_URL`, `QDRANT_CREDENTIALS_SECRET`, `QDRANT_CREDENTIALS_API_KEY_KEY` | when enabled |
| `PHOENIX_URL`, `PHOENIX_OTLP_GRPC_ENDPOINT`, `PHOENIX_OTLP_HTTP_ENDPOINT`, `PHOENIX_CREDENTIALS_SECRET`, `PHOENIX_CREDENTIALS_ADMIN_PASSWORD_KEY` | when enabled (admin user `admin@localhost`) |

### Spark Connect with a token

The Spark 4.1 Connect server has no TLS, and the PySpark client only sends `;token=` over TLS (plaintext only
to `localhost`). Pass the token as gRPC metadata instead, which the stock client supports:

```python
SparkSession.builder.remote(f"sc://{host}:15002/;authorization=Bearer {token}").getOrCreate()
```

`spark.auth.enabled=false` turns the token off (the NetworkPolicy still applies).

### Airflow task logs

Task logs are written to the `<fullname>-airflow-logs` PVC (2Gi, `airflow.logs.persistence`). With LocalExecutor, tasks
and the api-server run in the same pod and share this volume, so logs of finished runs stay readable in the UI and API
across pod restarts and upgrades. They are deleted with `helm uninstall`. No S3 remote logging is configured; that is
only needed when logs must outlive the release. Set `airflow.logs.persistence.enabled=false` for an emptyDir: logs then
survive task completion but not a pod restart. CI checks this: a DAG written into the DAG PVC runs, then its task log
is read through the API, the Airflow pod is restarted, and the same log is read again.

### Spark History Server

Spark Connect writes rolled event logs (`sparkHistory.rollingMaxFileSize`, default 10m) to `sparkHistory.eventLogDir`
(default `s3a://<bucket>/spark-events`). The History Server reads them on port 18080 and cleans up logs older than
`cleanerMaxAge`. The Connect server is one long-running Spark application, and S3 shows a file only once it is
closed. So the History Server sees that app up to its last rolled file, and completely after it stops: every pod
restart or upgrade stops it gracefully, which closes the log. For what is running now, use the live Spark UI on the
Connect server (port 4040), or the MCP's `live` server.

### MCP servers for agents

Both servers use MCP streamable HTTP and are off by default. Neither authenticates its callers, so only the
NetworkPolicy protects them: add agents through `networkPolicy.extraIngressFrom`.

- **Spark History MCP** (Kubeflow `mcp-apache-spark-history-server` 1.0.1) offers 17 tools: applications, jobs,
  stages, executors, SQL plans, bottleneck and comparison tools. It has two server names: `history` (default, the
  History Server) and `live` (the running Connect server's Spark UI; `sparkHistoryMcp.liveServer`). DNS-rebinding
  protection accepts only this release's Service names and `localhost` (for `kubectl port-forward`).
- **Airflow MCP** (Astronomer `astro-airflow-mcp` 0.9.1) offers 34 tools: DAGs, runs, task logs, diagnostics,
  variables, pools and assets. It logs in as its own Airflow user (`airflowMcp.username`, default `mcp`), never as
  the admin. With `readOnly: true` (default) that user has the `viewer` role and the MCP blocks write tools, so even
  a misbehaving client gets 403 from Airflow. With `readOnly: false` the user gets the `op` role (trigger, pause,
  clear). Telemetry to Astronomer is off.

Example client entry (from inside the release, or through `kubectl port-forward`):

```json
{"mcpServers": {"spark-history": {"type": "http", "url": "http://localhost:18888/mcp"},
                "airflow": {"type": "http", "url": "http://localhost:8000/mcp"}}}
```

### Delivering Airflow DAGs

- `airflow.dags.source=pvc` (default): DAGs are read from `airflow.dags.pvc.existingClaim`, or from the chart-made
  PVC `<fullname>-airflow-dags`. An external agent writes `.py` files into it. Use a `ReadWriteMany` class if the
  writer runs while Airflow is up on another node. New files are picked up within about 30 s.
- `airflow.dags.source=gitSync`: a git-sync sidecar pulls `airflow.dags.gitSync.repo`@`ref`. Credentials come
  from `airflow.dags.gitSync.existingSecret` (`username`/`password`, or `ssh-privatekey` + `known_hosts`).

## Credentials

There are no default passwords. Each credential takes an `existingSecret`. Otherwise the chart generates a
random value once and reuses it on every upgrade (`lookup` on the live Secret).

| Credential | Generated Secret | Keys | Bring your own |
|---|---|---|---|
| S3 access key pair (SeaweedFS identity, used by HMS/Spark/Trino) | `<fullname>-s3` | `access-key`, `secret-key` | `s3.auth.existingSecret` (+ key names) |
| HMS Postgres password | `<fullname>-hms-db` | `password` | `hms.database.auth.existingSecret` |
| Spark Connect token | `<fullname>-spark-connect` | `token` | `spark.auth.existingSecret` |
| Airflow admin password | `<fullname>-airflow` | `username`, `password` | `airflow.auth.existingSecret` |
| Airflow fernet key, JWT secret, API secret key | `<fullname>-airflow` | `fernet-key`, `jwt-secret`, `api-secret-key` | `airflow.secrets.existingSecret` |
| Airflow Postgres password (must be URL-safe) | `<fullname>-airflow-db` | `password` | `airflow.database.auth.existingSecret` |
| Airflow MCP user password (`viewer`/`op` role) | `<fullname>-airflow-mcp` | `password` | `airflowMcp.auth.existingSecret` |
| Jupyter token | `<fullname>-jupyter` | `token` | `jupyter.auth.existingSecret` |
| Qdrant API key | `<fullname>-qdrant` | `api-key` | `qdrant.auth.existingSecret` |
| Phoenix signing secret, admin password | `<fullname>-phoenix` | `secret`, `admin-password` | `phoenix.auth.existingSecret` |

> **`lookup` caveat.** `helm template`, `--dry-run=client`, Argo CD and Flux post-renderers cannot read live
> Secrets. They would generate new values on every render. Tools that render without cluster access must set
> every `existingSecret`. `helm install/upgrade` and the Helm Go SDK keep the values stable.

**Endpoints without authentication.** In this setup, HMS Thrift and Iceberg REST (`metastore.catalog.servlet.auth=none`),
Trino (password auth needs HTTPS and a shared secret), the Spark History Server, both MCP servers and Ollama (no
auth feature) are unauthenticated. With
`networkPolicy.enabled=true` (the default), only pods of the same release can reach any component. Add
consumers through `networkPolicy.extraIngressFrom`, for example:

```yaml
networkPolicy:
  extraIngressFrom:
    - namespaceSelector:
        matchLabels: {kubernetes.io/metadata.name: sandpiper}
```

## Multiple installs

- Every name is `<fullname>-<component>`, with `<fullname>` capped at 40 characters. Every object lives in
  the release namespace. Selectors include `app.kubernetes.io/instance`.
- Nothing is cluster-scoped. Spark executor RBAC is a namespaced Role/RoleBinding for pods, configmaps and
  persistentvolumeclaims.
- Executor pods are named `<fullname>-exec-*` and are owned by the Connect server pod, so they are deleted with it.
- Persistent data uses chart-managed PVCs (Deployments, not StatefulSets), so `helm uninstall` removes them.
  Set `*.persistence.enabled=false` for emptyDir-only, fully ephemeral labs.

## Security

Every pod and container (init containers and Spark executors included) runs with `runAsNonRoot`, a numeric
UID, `allowPrivilegeEscalation: false`, `capabilities.drop: [ALL]`, `seccompProfile: RuntimeDefault` and no
privileged mode, host namespaces, hostPath or hostPort. Pod Security Standard **restricted** is met by every
component. Services are `ClusterIP` only.

| Component | UID:GID | Notes |
|---|---|---|
| SeaweedFS | 1000:1000 | Image entrypoint only chowns when root, so that step is skipped |
| HMS | 1000:1000 | Entrypoint renders config into `/opt/hive/conf` (image layer), so the root FS is not read-only |
| Postgres (HMS, Airflow) | 999:999 | |
| Spark Connect + executors, Spark History Server | 185:185 | `/tmp` is an emptyDir (exec allowed): zstd event-log decoding loads a native library from it |
| Trino | 1000:1000 | |
| Airflow | 50000:0 | Official image convention (group 0) |
| Jupyter | 1000:100 | |
| Ollama | 1000:1000 | `HOME` and models on the data volume. **Not run-tested** (image too large for the authoring sandbox, no GPU in CI) |
| Qdrant | 1000:1000 | `-unprivileged` image |
| Phoenix | 65532:65532 | distroless `nonroot` image |
| Spark History MCP | 1000:1000 | official Kubeflow image |
| Airflow MCP | 1000:1000 | |

`readOnlyRootFilesystem` is not set (not required by "restricted"). HMS and Spark write to their image
directories at start-up.

## Resource footprint

Requests/limits are the chart defaults. "Idle RSS" was measured with the rendered configs (Docker, 2026-10-12).

| Component | CPU req | Mem req | Mem limit | Idle RSS | Persistence (default) |
|---|---|---|---|---|---|
| SeaweedFS | 100m | 256Mi | 1Gi | 160Mi | 10Gi |
| HMS | 200m | 1Gi | 1536Mi | 520Mi | — |
| HMS Postgres | 50m | 128Mi | 512Mi | 70–145Mi | 2Gi |
| Spark Connect server (driver) | 250m | 1536Mi | 2Gi | 890Mi | — |
| Spark executor (per pod, `master: kubernetes`) | 250m | 1408Mi (1g + 384m overhead) | same | — | — |
| Trino | 250m | 2Gi | 2Gi | 1.1Gi | — |
| Spark History Server | 100m | 768Mi | 1Gi | — | event logs in S3 |
| Spark History MCP | 50m | 128Mi | 512Mi | — | — |
| Airflow MCP | 50m | 128Mi | 512Mi | — | — |
| Airflow (api-server + scheduler + dag-processor) | 250m | 1408Mi | 3.5Gi | 910Mi | DAGs 1Gi, task logs 2Gi |
| Airflow Postgres | 50m | 128Mi | 512Mi | 90Mi | 2Gi |
| Jupyter | 100m | 512Mi | 4Gi | — | 5Gi |
| Ollama | 500m | 2Gi | 8Gi (+ GPU) | — | 20Gi |
| Qdrant | 100m | 256Mi | 2Gi | 55Mi | 5Gi |
| Phoenix | 100m | 512Mi | 2Gi | 420Mi | 2Gi |

Profiles (sum of requests):
- **minimal** (`ci/minimal-values.yaml`: hms, seaweedfs, spark, trino, 1 executor): ≈1.1 CPU, ≈6.3Gi.
- **default** (minimal + Airflow + Spark History Server): ≈1.5 CPU, ≈8.5Gi. Each MCP server adds 50m / 128Mi.

Start-up times measured per component on the rendered configs: SeaweedFS + bucket ≈10 s, HMS schema init on an
empty DB ≈30 s, Trino ≈60 s, Spark Connect ≈15 s, Airflow ≈30 s (migrate + api-server). Components start in
parallel; only HMS waits (for Postgres and the bucket). CI asserts that the minimal profile is ready in under
3 minutes on kind, with images pre-loaded.

## Requirements

Kubernetes: `>=1.29.0-0`

## Values

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| global.imageRegistry | string | `""` | Registry that replaces every image's `registry` (for mirrors / air-gapped clusters). Empty keeps each image's own registry. |
| global.imagePullSecrets | list | `[]` | Pull secrets (names of existing `kubernetes.io/dockerconfigjson` Secrets) added to every pod. |
| global.storageClass | string | `""` | Default StorageClass for every PVC. Empty uses the cluster default; `-` sets `storageClassName: ""`. |
| global.runtimeClassName | string | `""` | Default RuntimeClass for every pod (for example `gvisor`). A component's own `runtimeClassName` wins. |
| global.podLabels | object | `{}` | Labels added to every pod. |
| global.podAnnotations | object | `{}` | Annotations added to every pod. |
| global.nodeSelector | object | `{}` | Default nodeSelector for every pod. A component's own `nodeSelector` replaces it when non-empty. |
| global.tolerations | list | `[]` | Default tolerations for every pod. A component's own `tolerations` replace them when non-empty. |
| global.containerSecurityContext | object | `{"allowPrivilegeEscalation":false,"capabilities":{"drop":["ALL"]},"privileged":false,"runAsNonRoot":true,"seccompProfile":{"type":"RuntimeDefault"}}` | Container securityContext applied to every container (merged under each component's `securityContext`). Meets Pod Security Standard "restricted". |
| nameOverride | string | `""` | Override the chart name part of resource names. |
| fullnameOverride | string | `""` | Override the full release name prefix used by every resource (max 40 characters recommended). |
| commonLabels | object | `{}` | Labels added to every resource. |
| networkPolicy.enabled | bool | `true` | Create a NetworkPolicy that only admits ingress from pods of this release (same namespace) plus `extraIngressFrom`. HMS, Trino and Ollama have no authentication in this setup (see the chart README), so keep this on in shared clusters. |
| networkPolicy.extraIngressFrom | list | `[]` | Extra NetworkPolicyPeer entries allowed to reach every component (for example the consumer's namespace). |
| s3.endpoint | string | `""` | External S3 endpoint URL. Leave empty to use the bundled SeaweedFS (requires `seaweedfs.enabled`). |
| s3.bucket | string | `"warehouse"` | Bucket holding the warehouse. The bundled SeaweedFS creates it; an external bucket must already exist. |
| s3.region | string | `"us-east-1"` | S3 region. |
| s3.pathStyleAccess | bool | `true` | Use path-style addressing (required for SeaweedFS). |
| s3.auth.existingSecret | string | `""` | Existing Secret holding the S3 access key pair. Empty: the chart generates `<fullname>-s3`. |
| s3.auth.accessKeyKey | string | `"access-key"` | Key in the Secret holding the access key ID. |
| s3.auth.secretKeyKey | string | `"secret-key"` | Key in the Secret holding the secret access key. |
| seaweedfs.enabled | bool | `true` | Deploy SeaweedFS (single-node master + volume + filer + S3 gateway). |
| seaweedfs.image.registry | string | `"docker.io"` | Image registry. |
| seaweedfs.image.repository | string | `"chrislusf/seaweedfs"` | Image repository. |
| seaweedfs.image.tag | string | `"4.48"` | Image tag. |
| seaweedfs.image.pullPolicy | string | `"IfNotPresent"` | Image pull policy. |
| seaweedfs.volumeMax | int | `64` | Maximum number of volumes. Must stay explicit: auto-sizing from free disk can leave no writable volume (S3 PUT 500). |
| seaweedfs.volumeSizeLimitMB | int | `1024` | Size limit of one volume, in MB. |
| seaweedfs.persistence.enabled | bool | `true` | Use a PVC for `/data`. `false` uses an emptyDir (data is lost with the pod). |
| seaweedfs.persistence.size | string | `"10Gi"` | PVC size. |
| seaweedfs.persistence.storageClass | string | `""` | StorageClass for this PVC. Empty falls back to `global.storageClass`; `-` sets `storageClassName: ""`. |
| seaweedfs.persistence.accessModes | list | `["ReadWriteOnce"]` | PVC access modes. |
| seaweedfs.resources | object | `{"limits":{"memory":"1Gi"},"requests":{"cpu":"100m","memory":"256Mi"}}` | Container resources. |
| seaweedfs.podSecurityContext | object | `{"fsGroup":1000,"runAsGroup":1000,"runAsNonRoot":true,"runAsUser":1000,"seccompProfile":{"type":"RuntimeDefault"}}` | Pod securityContext. |
| seaweedfs.securityContext | object | `{}` | Container securityContext (merged over `global.containerSecurityContext`). |
| seaweedfs.runtimeClassName | string | `""` | RuntimeClass for the pod. |
| seaweedfs.nodeSelector | object | `{}` | nodeSelector for the pod. |
| seaweedfs.tolerations | list | `[]` | Tolerations for the pod. |
| seaweedfs.affinity | object | `{}` | Affinity for the pod. |
| seaweedfs.podLabels | object | `{}` | Extra pod labels. |
| seaweedfs.podAnnotations | object | `{}` | Extra pod annotations. |
| hms.enabled | bool | `true` | Deploy the Hive Metastore (Thrift on 9083, built-in Iceberg REST catalog on 9084) and its Postgres. |
| hms.image.registry | string | `"ghcr.io"` | Image registry. |
| hms.image.repository | string | `"tarekabouzeid/data-lab-playground/hive-metastore"` | Image repository (built from `hive-metastore/Dockerfile`, published by the release workflow). |
| hms.image.tag | string | `"3.0.0"` | Image tag (lab release; HMS 4.2.1 inside). |
| hms.image.pullPolicy | string | `"IfNotPresent"` | Image pull policy. |
| hms.heapSize | string | `"1G"` | JVM max heap passed as `-Xmx`. |
| hms.resources | object | `{"limits":{"memory":"1536Mi"},"requests":{"cpu":"200m","memory":"1Gi"}}` | Container resources. |
| hms.podSecurityContext | object | `{"fsGroup":1000,"runAsGroup":1000,"runAsNonRoot":true,"runAsUser":1000,"seccompProfile":{"type":"RuntimeDefault"}}` | Pod securityContext. |
| hms.securityContext | object | `{}` | Container securityContext (merged over `global.containerSecurityContext`). |
| hms.runtimeClassName | string | `""` | RuntimeClass for the pod. |
| hms.nodeSelector | object | `{}` | nodeSelector for the pod. |
| hms.tolerations | list | `[]` | Tolerations for the pod. |
| hms.affinity | object | `{}` | Affinity for the pod. |
| hms.podLabels | object | `{}` | Extra pod labels. |
| hms.podAnnotations | object | `{}` | Extra pod annotations. |
| hms.extraProperties | object | `{}` | Extra `<property>` entries appended to metastore-site.xml (name -> value). |
| hms.database.image.registry | string | `"docker.io"` | Image registry. |
| hms.database.image.repository | string | `"postgres"` | Image repository. |
| hms.database.image.tag | string | `"17.11"` | Image tag. |
| hms.database.image.pullPolicy | string | `"IfNotPresent"` | Image pull policy. |
| hms.database.name | string | `"metastore"` | Database name. |
| hms.database.username | string | `"hive"` | Database user. |
| hms.database.auth.existingSecret | string | `""` | Existing Secret holding the database password. Empty: the chart generates `<fullname>-hms-db`. |
| hms.database.auth.passwordKey | string | `"password"` | Key in the Secret holding the password. |
| hms.database.persistence.enabled | bool | `true` | Use a PVC for the database. `false` uses an emptyDir. |
| hms.database.persistence.size | string | `"2Gi"` | PVC size. |
| hms.database.persistence.storageClass | string | `""` | StorageClass for this PVC. Empty falls back to `global.storageClass`; `-` sets `storageClassName: ""`. |
| hms.database.persistence.accessModes | list | `["ReadWriteOnce"]` | PVC access modes. |
| hms.database.resources | object | `{"limits":{"memory":"512Mi"},"requests":{"cpu":"50m","memory":"128Mi"}}` | Container resources. |
| hms.database.podSecurityContext | object | `{"fsGroup":999,"runAsGroup":999,"runAsNonRoot":true,"runAsUser":999,"seccompProfile":{"type":"RuntimeDefault"}}` | Pod securityContext. |
| hms.database.securityContext | object | `{}` | Container securityContext (merged over `global.containerSecurityContext`). |
| hms.database.runtimeClassName | string | `""` | RuntimeClass for the pod. |
| hms.database.nodeSelector | object | `{}` | nodeSelector for the pod. |
| hms.database.tolerations | list | `[]` | Tolerations for the pod. |
| hms.database.affinity | object | `{}` | Affinity for the pod. |
| hms.database.podLabels | object | `{}` | Extra pod labels. |
| hms.database.podAnnotations | object | `{}` | Extra pod annotations. |
| spark.enabled | bool | `true` | Deploy a Spark Connect server (gRPC 15002). Requires `hms.enabled`. |
| spark.image.registry | string | `"ghcr.io"` | Image registry. |
| spark.image.repository | string | `"tarekabouzeid/data-lab-playground/spark"` | Image repository (built from `spark/Dockerfile`; also used for executors and `helm test`). |
| spark.image.tag | string | `"3.0.0"` | Image tag (lab release; Spark 4.1.3 + Iceberg 1.12.0 inside). |
| spark.image.pullPolicy | string | `"IfNotPresent"` | Image pull policy. |
| spark.master | string | `"kubernetes"` | Where executors run: `kubernetes` (executor pods in the release namespace, namespaced RBAC) or `local` (threads inside the server pod, no RBAC). |
| spark.auth.enabled | bool | `true` | Require a bearer token from Spark Connect clients (`sc://host:15002/;token=...`). |
| spark.auth.existingSecret | string | `""` | Existing Secret holding the token. Empty: the chart generates `<fullname>-spark-connect`. |
| spark.auth.tokenKey | string | `"token"` | Key in the Secret holding the token. |
| spark.driver.memory | string | `"1g"` | Driver (Connect server) JVM memory. |
| spark.driver.resources | object | `{"limits":{"memory":"2Gi"},"requests":{"cpu":"250m","memory":"1536Mi"}}` | Driver (Connect server) container resources. |
| spark.executor.instances | int | `1` | Number of executor pods (`master: kubernetes`) or threads (`master: local`). |
| spark.executor.cores | int | `1` | Cores per executor. |
| spark.executor.memory | string | `"1g"` | Executor JVM memory (`spark.executor.memory`). Pod memory adds `spark.executor.memoryOverhead` (default 384m). |
| spark.executor.requestCores | string | `"250m"` | CPU request per executor pod (`spark.kubernetes.executor.request.cores`). |
| spark.executor.limitCores | string | `""` | CPU limit per executor pod. Empty sets none. |
| spark.icebergCatalogName | string | `"iceberg_catalog"` | Name of the Iceberg catalog registered in Spark (points at the HMS Iceberg REST catalog). |
| spark.extraConf | object | `{}` | Extra Spark properties written to spark-defaults.conf. |
| spark.serviceAccount.create | bool | `true` | Create a ServiceAccount for the Connect server (it creates executor pods when `master: kubernetes`). |
| spark.serviceAccount.name | string | `""` | ServiceAccount name. Empty: `<fullname>-spark`. |
| spark.rbac.create | bool | `true` | Create a namespaced Role/RoleBinding letting the Connect server manage executor pods, ConfigMaps and PVCs. |
| spark.podSecurityContext | object | `{"fsGroup":185,"runAsGroup":185,"runAsNonRoot":true,"runAsUser":185,"seccompProfile":{"type":"RuntimeDefault"}}` | Pod securityContext (Connect server and executors). |
| spark.securityContext | object | `{}` | Container securityContext (merged over `global.containerSecurityContext`). |
| spark.runtimeClassName | string | `""` | RuntimeClass for the server and executor pods. |
| spark.nodeSelector | object | `{}` | nodeSelector for the server and executor pods. |
| spark.tolerations | list | `[]` | Tolerations for the server and executor pods. |
| spark.affinity | object | `{}` | Affinity for the Connect server pod. |
| spark.podLabels | object | `{}` | Extra labels for the server and executor pods. |
| spark.podAnnotations | object | `{}` | Extra annotations for the server and executor pods. |
| sparkHistory.enabled | bool | `true` | Deploy the Spark History Server (UI + REST on 18080). Turns on Spark event logging to `eventLogDir`. Requires `spark.enabled`. Uses `spark.image`. A running Spark Connect server appears here only after it stops or rolls a log file (S3 shows a file only once it is closed); use the live Spark UI (port 4040) for the running server. |
| sparkHistory.eventLogDir | string | `""` | Event log location written by Spark and read by the History Server. Empty: `s3a://<s3.bucket>/spark-events`. |
| sparkHistory.rollingMaxFileSize | string | `"10m"` | Roll the Spark Connect server's event log at this size so a long-running server shows up in history (Spark minimum: 10m). |
| sparkHistory.cleanerMaxAge | string | `"7d"` | Delete event logs older than this. |
| sparkHistory.heapSize | string | `"512m"` | JVM heap of the History Server (`SPARK_DAEMON_MEMORY`). |
| sparkHistory.resources | object | `{"limits":{"memory":"1Gi"},"requests":{"cpu":"100m","memory":"768Mi"}}` | Container resources. |
| sparkHistory.podSecurityContext | object | `{"fsGroup":185,"runAsGroup":185,"runAsNonRoot":true,"runAsUser":185,"seccompProfile":{"type":"RuntimeDefault"}}` | Pod securityContext. |
| sparkHistory.securityContext | object | `{}` | Container securityContext (merged over `global.containerSecurityContext`). |
| sparkHistory.runtimeClassName | string | `""` | RuntimeClass for the pod. |
| sparkHistory.nodeSelector | object | `{}` | nodeSelector for the pod. |
| sparkHistory.tolerations | list | `[]` | Tolerations for the pod. |
| sparkHistory.affinity | object | `{}` | Affinity for the pod. |
| sparkHistory.podLabels | object | `{}` | Extra pod labels. |
| sparkHistory.podAnnotations | object | `{}` | Extra pod annotations. |
| sparkHistoryMcp.enabled | bool | `false` | Deploy the Kubeflow Spark History Server MCP server (streamable HTTP on 18888, path `/mcp`). No caller authentication: reachable only through the NetworkPolicy. Requires `sparkHistory.enabled`. Queries the History Server (`history`, default) and, when Spark is enabled, the live Spark Connect UI (`live`). |
| sparkHistoryMcp.image.registry | string | `"ghcr.io"` | Image registry. |
| sparkHistoryMcp.image.repository | string | `"kubeflow/mcp-apache-spark-history-server"` | Image repository (official Kubeflow image). |
| sparkHistoryMcp.image.tag | string | `"1.0.1"` | Image tag. |
| sparkHistoryMcp.image.pullPolicy | string | `"IfNotPresent"` | Image pull policy. |
| sparkHistoryMcp.liveServer | bool | `true` | Also query the live Spark UI of the running Spark Connect server (MCP server name `live`); `history` is always the default. |
| sparkHistoryMcp.timeout | int | `30` | HTTP timeout in seconds for calls to the History Server / Spark UI. |
| sparkHistoryMcp.includePlanDescription | bool | `false` | Include SQL plan descriptions in SQL tool results (larger responses, more detail for query tuning). |
| sparkHistoryMcp.debug | bool | `false` | Verbose MCP server logging. |
| sparkHistoryMcp.extraEnv | list | `[]` | Extra environment variables for the MCP container (list of EnvVar). |
| sparkHistoryMcp.resources | object | `{"limits":{"memory":"512Mi"},"requests":{"cpu":"50m","memory":"128Mi"}}` | Container resources. |
| sparkHistoryMcp.podSecurityContext | object | `{"fsGroup":1000,"runAsGroup":1000,"runAsNonRoot":true,"runAsUser":1000,"seccompProfile":{"type":"RuntimeDefault"}}` | Pod securityContext. |
| sparkHistoryMcp.securityContext | object | `{}` | Container securityContext (merged over `global.containerSecurityContext`). |
| sparkHistoryMcp.runtimeClassName | string | `""` | RuntimeClass for the pod. |
| sparkHistoryMcp.nodeSelector | object | `{}` | nodeSelector for the pod. |
| sparkHistoryMcp.tolerations | list | `[]` | Tolerations for the pod. |
| sparkHistoryMcp.affinity | object | `{}` | Affinity for the pod. |
| sparkHistoryMcp.podLabels | object | `{}` | Extra pod labels. |
| sparkHistoryMcp.podAnnotations | object | `{}` | Extra pod annotations. |
| airflowMcp.enabled | bool | `false` | Deploy the Astronomer Airflow MCP server (streamable HTTP on 8000, path `/mcp`). Logs in to Airflow as its own least-privilege user (`username`). No caller authentication: reachable only through the NetworkPolicy. Requires `airflow.enabled`. |
| airflowMcp.image.registry | string | `"ghcr.io"` | Image registry. |
| airflowMcp.image.repository | string | `"tarekabouzeid/data-lab-playground/airflow-mcp"` | Image repository (built from `airflow-mcp/Dockerfile`: astro-airflow-mcp 0.9.1, published by the release workflow). |
| airflowMcp.image.tag | string | `"3.0.0"` | Image tag (lab release). |
| airflowMcp.image.pullPolicy | string | `"IfNotPresent"` | Image pull policy. |
| airflowMcp.readOnly | bool | `true` | Read-only mode. `true`: the MCP's Airflow user gets the `viewer` role and the MCP blocks write tools, so Airflow itself rejects writes too. `false`: the user gets the `op` role (trigger, pause, clear; no admin settings). |
| airflowMcp.username | string | `"mcp"` | Dedicated Airflow user the MCP logs in as (added to Airflow's SimpleAuthManager users). |
| airflowMcp.auth.existingSecret | string | `""` | Existing Secret holding the MCP user's Airflow password. Empty: the chart generates `<fullname>-airflow-mcp`. |
| airflowMcp.auth.passwordKey | string | `"password"` | Key in the Secret holding the password. |
| airflowMcp.extraEnv | list | `[]` | Extra environment variables for the MCP container (list of EnvVar). |
| airflowMcp.resources | object | `{"limits":{"memory":"512Mi"},"requests":{"cpu":"50m","memory":"128Mi"}}` | Container resources. |
| airflowMcp.podSecurityContext | object | `{"fsGroup":1000,"runAsGroup":1000,"runAsNonRoot":true,"runAsUser":1000,"seccompProfile":{"type":"RuntimeDefault"}}` | Pod securityContext. |
| airflowMcp.securityContext | object | `{}` | Container securityContext (merged over `global.containerSecurityContext`). |
| airflowMcp.runtimeClassName | string | `""` | RuntimeClass for the pod. |
| airflowMcp.nodeSelector | object | `{}` | nodeSelector for the pod. |
| airflowMcp.tolerations | list | `[]` | Tolerations for the pod. |
| airflowMcp.affinity | object | `{}` | Affinity for the pod. |
| airflowMcp.podLabels | object | `{}` | Extra pod labels. |
| airflowMcp.podAnnotations | object | `{}` | Extra pod annotations. |
| trino.enabled | bool | `true` | Deploy a single-node Trino (coordinator also schedules work). Requires `hms.enabled`. |
| trino.image.registry | string | `"docker.io"` | Image registry. |
| trino.image.repository | string | `"trinodb/trino"` | Image repository. |
| trino.image.tag | string | `"483"` | Image tag. |
| trino.image.pullPolicy | string | `"IfNotPresent"` | Image pull policy. |
| trino.heapSize | string | `"1536M"` | JVM max heap (`-Xmx`). Keep it about 75% of the memory limit. |
| trino.extraConfig | object | `{}` | Extra lines for config.properties (name -> value). |
| trino.additionalCatalogs | object | `{}` | Extra catalogs: catalog name -> properties file content. `hive`, `iceberg` and `lakehouse` are always created. |
| trino.resources | object | `{"limits":{"memory":"2Gi"},"requests":{"cpu":"250m","memory":"2Gi"}}` | Container resources. |
| trino.podSecurityContext | object | `{"fsGroup":1000,"runAsGroup":1000,"runAsNonRoot":true,"runAsUser":1000,"seccompProfile":{"type":"RuntimeDefault"}}` | Pod securityContext. |
| trino.securityContext | object | `{}` | Container securityContext (merged over `global.containerSecurityContext`). |
| trino.runtimeClassName | string | `""` | RuntimeClass for the pod. |
| trino.nodeSelector | object | `{}` | nodeSelector for the pod. |
| trino.tolerations | list | `[]` | Tolerations for the pod. |
| trino.affinity | object | `{}` | Affinity for the pod. |
| trino.podLabels | object | `{}` | Extra pod labels. |
| trino.podAnnotations | object | `{}` | Extra pod annotations. |
| airflow.enabled | bool | `true` | Deploy Airflow 3 (LocalExecutor) in one pod: api-server, scheduler, dag-processor (+ optional triggerer), with its own Postgres. |
| airflow.image.registry | string | `"docker.io"` | Image registry. |
| airflow.image.repository | string | `"apache/airflow"` | Image repository. |
| airflow.image.tag | string | `"3.3.2-python3.12"` | Image tag. |
| airflow.image.pullPolicy | string | `"IfNotPresent"` | Image pull policy. |
| airflow.auth.username | string | `"admin"` | Admin username for the API / UI (SimpleAuthManager). |
| airflow.auth.existingSecret | string | `""` | Existing Secret holding the admin password. Empty: the chart generates `<fullname>-airflow`. |
| airflow.auth.passwordKey | string | `"password"` | Key in the Secret holding the admin password. |
| airflow.secrets.existingSecret | string | `""` | Existing Secret holding `fernet-key`, `jwt-secret` and `api-secret-key`. Empty: the chart generates them in `<fullname>-airflow`. |
| airflow.triggerer.enabled | bool | `false` | Run the triggerer (needed for deferrable operators). |
| airflow.dags.source | string | `"pvc"` | Where DAG files come from: `pvc`, `gitSync` or `none` (empty dir). |
| airflow.dags.subPath | string | `""` | Path inside the DAG source that holds the DAG files (git-sync: inside the repo). |
| airflow.dags.pvc.existingClaim | string | `""` | Existing PVC with the DAGs (an external agent writes to it). Empty: the chart creates `<fullname>-airflow-dags`. |
| airflow.dags.pvc.size | string | `"1Gi"` | Size of the chart-created PVC. |
| airflow.dags.pvc.storageClass | string | `""` | StorageClass of the chart-created PVC. Empty falls back to `global.storageClass`; `-` sets `storageClassName: ""`. |
| airflow.dags.pvc.accessModes | list | `["ReadWriteOnce"]` | Access modes of the chart-created PVC. Use `ReadWriteMany` if another pod writes DAGs while Airflow runs. |
| airflow.dags.gitSync.image.registry | string | `"registry.k8s.io"` | Image registry. |
| airflow.dags.gitSync.image.repository | string | `"git-sync/git-sync"` | Image repository. |
| airflow.dags.gitSync.image.tag | string | `"v4.7.1"` | Image tag. |
| airflow.dags.gitSync.image.pullPolicy | string | `"IfNotPresent"` | Image pull policy. |
| airflow.dags.gitSync.repo | string | `""` | Git repository URL (required when `dags.source: gitSync`). |
| airflow.dags.gitSync.ref | string | `"main"` | Branch, tag or commit to sync. |
| airflow.dags.gitSync.period | string | `"60s"` | Sync period. |
| airflow.dags.gitSync.depth | int | `1` | Clone depth. |
| airflow.dags.gitSync.existingSecret | string | `""` | Existing Secret with git credentials: keys `username`/`password` (HTTPS) or `ssh-privatekey` (+ optional `known_hosts`). |
| airflow.dags.gitSync.resources | object | `{"limits":{"memory":"128Mi"},"requests":{"cpu":"10m","memory":"32Mi"}}` | Container resources. |
| airflow.logs.persistence.enabled | bool | `true` | Keep task logs on a PVC so they stay viewable in the UI after pod restarts (LocalExecutor: tasks and api-server share this volume). `false` uses an emptyDir: logs survive task completion but not a pod restart. |
| airflow.logs.persistence.size | string | `"2Gi"` | PVC size. |
| airflow.logs.persistence.storageClass | string | `""` | StorageClass for this PVC. Empty falls back to `global.storageClass`; `-` sets `storageClassName: ""`. |
| airflow.logs.persistence.accessModes | list | `["ReadWriteOnce"]` | PVC access modes. |
| airflow.extraEnv | list | `[]` | Extra environment variables for every Airflow container (list of EnvVar). |
| airflow.resources | object | `{"apiServer":{"limits":{"memory":"1Gi"},"requests":{"cpu":"100m","memory":"512Mi"}},"dagProcessor":{"limits":{"memory":"1Gi"},"requests":{"cpu":"50m","memory":"384Mi"}},"scheduler":{"limits":{"memory":"1536Mi"},"requests":{"cpu":"100m","memory":"512Mi"}},"triggerer":{"limits":{"memory":"768Mi"},"requests":{"cpu":"50m","memory":"256Mi"}}}` | Resources of each Airflow container (api-server, scheduler, dag-processor, triggerer). |
| airflow.podSecurityContext | object | `{"fsGroup":0,"runAsGroup":0,"runAsNonRoot":true,"runAsUser":50000,"seccompProfile":{"type":"RuntimeDefault"}}` | Pod securityContext (official image: uid 50000, gid 0). |
| airflow.securityContext | object | `{}` | Container securityContext (merged over `global.containerSecurityContext`). |
| airflow.runtimeClassName | string | `""` | RuntimeClass for the pod. |
| airflow.nodeSelector | object | `{}` | nodeSelector for the pod. |
| airflow.tolerations | list | `[]` | Tolerations for the pod. |
| airflow.affinity | object | `{}` | Affinity for the pod. |
| airflow.podLabels | object | `{}` | Extra pod labels. |
| airflow.podAnnotations | object | `{}` | Extra pod annotations. |
| airflow.database.image.registry | string | `"docker.io"` | Image registry. |
| airflow.database.image.repository | string | `"postgres"` | Image repository. |
| airflow.database.image.tag | string | `"17.11"` | Image tag. |
| airflow.database.image.pullPolicy | string | `"IfNotPresent"` | Image pull policy. |
| airflow.database.name | string | `"airflow"` | Database name. |
| airflow.database.username | string | `"airflow"` | Database user. |
| airflow.database.auth.existingSecret | string | `""` | Existing Secret holding the database password (must be URL-safe). Empty: the chart generates `<fullname>-airflow-db`. |
| airflow.database.auth.passwordKey | string | `"password"` | Key in the Secret holding the password. |
| airflow.database.persistence.enabled | bool | `true` | Use a PVC for the database. `false` uses an emptyDir. |
| airflow.database.persistence.size | string | `"2Gi"` | PVC size. |
| airflow.database.persistence.storageClass | string | `""` | StorageClass for this PVC. Empty falls back to `global.storageClass`; `-` sets `storageClassName: ""`. |
| airflow.database.persistence.accessModes | list | `["ReadWriteOnce"]` | PVC access modes. |
| airflow.database.resources | object | `{"limits":{"memory":"512Mi"},"requests":{"cpu":"50m","memory":"128Mi"}}` | Container resources. |
| airflow.database.podSecurityContext | object | `{"fsGroup":999,"runAsGroup":999,"runAsNonRoot":true,"runAsUser":999,"seccompProfile":{"type":"RuntimeDefault"}}` | Pod securityContext. |
| airflow.database.securityContext | object | `{}` | Container securityContext (merged over `global.containerSecurityContext`). |
| airflow.database.runtimeClassName | string | `""` | RuntimeClass for the pod. |
| airflow.database.nodeSelector | object | `{}` | nodeSelector for the pod. |
| airflow.database.tolerations | list | `[]` | Tolerations for the pod. |
| airflow.database.affinity | object | `{}` | Affinity for the pod. |
| airflow.database.podLabels | object | `{}` | Extra pod labels. |
| airflow.database.podAnnotations | object | `{}` | Extra pod annotations. |
| jupyter.enabled | bool | `false` | Deploy JupyterLab (connects to Spark Connect, Trino and the GenAI services that are enabled). |
| jupyter.image.registry | string | `"ghcr.io"` | Image registry. |
| jupyter.image.repository | string | `"tarekabouzeid/data-lab-playground/jupyter"` | Image repository (built from `jupyter/Dockerfile`, published by the release workflow). |
| jupyter.image.tag | string | `"3.0.0"` | Image tag (lab release). |
| jupyter.image.pullPolicy | string | `"IfNotPresent"` | Image pull policy. |
| jupyter.auth.existingSecret | string | `""` | Existing Secret holding the Jupyter token. Empty: the chart generates `<fullname>-jupyter`. |
| jupyter.auth.tokenKey | string | `"token"` | Key in the Secret holding the token. |
| jupyter.persistence.enabled | bool | `true` | Use a PVC for `/home/jovyan/work`. `false` uses an emptyDir. |
| jupyter.persistence.size | string | `"5Gi"` | PVC size. |
| jupyter.persistence.storageClass | string | `""` | StorageClass for this PVC. Empty falls back to `global.storageClass`; `-` sets `storageClassName: ""`. |
| jupyter.persistence.accessModes | list | `["ReadWriteOnce"]` | PVC access modes. |
| jupyter.resources | object | `{"limits":{"memory":"4Gi"},"requests":{"cpu":"100m","memory":"512Mi"}}` | Container resources. |
| jupyter.podSecurityContext | object | `{"fsGroup":100,"runAsGroup":100,"runAsNonRoot":true,"runAsUser":1000,"seccompProfile":{"type":"RuntimeDefault"}}` | Pod securityContext (jovyan: uid 1000, gid 100). |
| jupyter.securityContext | object | `{}` | Container securityContext (merged over `global.containerSecurityContext`). |
| jupyter.runtimeClassName | string | `""` | RuntimeClass for the pod. |
| jupyter.nodeSelector | object | `{}` | nodeSelector for the pod. |
| jupyter.tolerations | list | `[]` | Tolerations for the pod. |
| jupyter.affinity | object | `{}` | Affinity for the pod. |
| jupyter.podLabels | object | `{}` | Extra pod labels. |
| jupyter.podAnnotations | object | `{}` | Extra pod annotations. |
| ollama.enabled | bool | `false` | Deploy Ollama (local LLM server, no authentication). Needs a GPU to be useful; CPU works but is slow. |
| ollama.image.registry | string | `"docker.io"` | Image registry. |
| ollama.image.repository | string | `"ollama/ollama"` | Image repository. |
| ollama.image.tag | string | `"0.40.3"` | Image tag. |
| ollama.image.pullPolicy | string | `"IfNotPresent"` | Image pull policy. |
| ollama.models | list | `[]` | Models pulled at startup (for example `llama3.2:3b`). Pod readiness waits for the pulls. |
| ollama.gpu.enabled | bool | `false` | Request GPUs. |
| ollama.gpu.resourceName | string | `"nvidia.com/gpu"` | Extended resource name of the GPU. |
| ollama.gpu.count | int | `1` | Number of GPUs. |
| ollama.persistence.enabled | bool | `true` | Use a PVC for models. `false` uses an emptyDir. |
| ollama.persistence.size | string | `"20Gi"` | PVC size. |
| ollama.persistence.storageClass | string | `""` | StorageClass for this PVC. Empty falls back to `global.storageClass`; `-` sets `storageClassName: ""`. |
| ollama.persistence.accessModes | list | `["ReadWriteOnce"]` | PVC access modes. |
| ollama.resources | object | `{"limits":{"memory":"8Gi"},"requests":{"cpu":"500m","memory":"2Gi"}}` | Container resources. |
| ollama.podSecurityContext | object | `{"fsGroup":1000,"runAsGroup":1000,"runAsNonRoot":true,"runAsUser":1000,"seccompProfile":{"type":"RuntimeDefault"}}` | Pod securityContext. |
| ollama.securityContext | object | `{}` | Container securityContext (merged over `global.containerSecurityContext`). |
| ollama.runtimeClassName | string | `""` | RuntimeClass for the pod (for example `nvidia`). |
| ollama.nodeSelector | object | `{}` | nodeSelector for the pod. |
| ollama.tolerations | list | `[]` | Tolerations for the pod. |
| ollama.affinity | object | `{}` | Affinity for the pod. |
| ollama.podLabels | object | `{}` | Extra pod labels. |
| ollama.podAnnotations | object | `{}` | Extra pod annotations. |
| qdrant.enabled | bool | `false` | Deploy Qdrant (vector database, API-key protected). |
| qdrant.image.registry | string | `"docker.io"` | Image registry. |
| qdrant.image.repository | string | `"qdrant/qdrant"` | Image repository. |
| qdrant.image.tag | string | `"v1.19.2-unprivileged"` | Image tag (the `-unprivileged` variant runs as uid 1000). |
| qdrant.image.pullPolicy | string | `"IfNotPresent"` | Image pull policy. |
| qdrant.auth.existingSecret | string | `""` | Existing Secret holding the API key. Empty: the chart generates `<fullname>-qdrant`. |
| qdrant.auth.apiKeyKey | string | `"api-key"` | Key in the Secret holding the API key. |
| qdrant.persistence.enabled | bool | `true` | Use a PVC for `/qdrant/storage`. `false` uses an emptyDir. |
| qdrant.persistence.size | string | `"5Gi"` | PVC size. |
| qdrant.persistence.storageClass | string | `""` | StorageClass for this PVC. Empty falls back to `global.storageClass`; `-` sets `storageClassName: ""`. |
| qdrant.persistence.accessModes | list | `["ReadWriteOnce"]` | PVC access modes. |
| qdrant.resources | object | `{"limits":{"memory":"2Gi"},"requests":{"cpu":"100m","memory":"256Mi"}}` | Container resources. |
| qdrant.podSecurityContext | object | `{"fsGroup":1000,"runAsGroup":1000,"runAsNonRoot":true,"runAsUser":1000,"seccompProfile":{"type":"RuntimeDefault"}}` | Pod securityContext. |
| qdrant.securityContext | object | `{}` | Container securityContext (merged over `global.containerSecurityContext`). |
| qdrant.runtimeClassName | string | `""` | RuntimeClass for the pod. |
| qdrant.nodeSelector | object | `{}` | nodeSelector for the pod. |
| qdrant.tolerations | list | `[]` | Tolerations for the pod. |
| qdrant.affinity | object | `{}` | Affinity for the pod. |
| qdrant.podLabels | object | `{}` | Extra pod labels. |
| qdrant.podAnnotations | object | `{}` | Extra pod annotations. |
| phoenix.enabled | bool | `false` | Deploy Arize Phoenix (LLM tracing; OTLP gRPC 4317, UI + OTLP HTTP 6006). SQLite storage. |
| phoenix.image.registry | string | `"docker.io"` | Image registry. |
| phoenix.image.repository | string | `"arizephoenix/phoenix"` | Image repository. |
| phoenix.image.tag | string | `"version-20.21.0-nonroot"` | Image tag. |
| phoenix.image.pullPolicy | string | `"IfNotPresent"` | Image pull policy. |
| phoenix.auth.enabled | bool | `true` | Enable Phoenix authentication (admin user + API keys). |
| phoenix.auth.existingSecret | string | `""` | Existing Secret holding `secret` (token signing, >= 32 chars) and `admin-password`. Empty: the chart generates `<fullname>-phoenix`. |
| phoenix.persistence.enabled | bool | `true` | Use a PVC for the SQLite working dir. `false` uses an emptyDir. |
| phoenix.persistence.size | string | `"2Gi"` | PVC size. |
| phoenix.persistence.storageClass | string | `""` | StorageClass for this PVC. Empty falls back to `global.storageClass`; `-` sets `storageClassName: ""`. |
| phoenix.persistence.accessModes | list | `["ReadWriteOnce"]` | PVC access modes. |
| phoenix.resources | object | `{"limits":{"memory":"2Gi"},"requests":{"cpu":"100m","memory":"512Mi"}}` | Container resources. |
| phoenix.podSecurityContext | object | `{"fsGroup":65532,"runAsGroup":65532,"runAsNonRoot":true,"runAsUser":65532,"seccompProfile":{"type":"RuntimeDefault"}}` | Pod securityContext (distroless nonroot: uid 65532). |
| phoenix.securityContext | object | `{}` | Container securityContext (merged over `global.containerSecurityContext`). |
| phoenix.runtimeClassName | string | `""` | RuntimeClass for the pod. |
| phoenix.nodeSelector | object | `{}` | nodeSelector for the pod. |
| phoenix.tolerations | list | `[]` | Tolerations for the pod. |
| phoenix.affinity | object | `{}` | Affinity for the pod. |
| phoenix.podLabels | object | `{}` | Extra pod labels. |
| phoenix.podAnnotations | object | `{}` | Extra pod annotations. |
| tests.enabled | bool | `true` | Render the `helm test` pod (Spark writes an Iceberg table through Spark Connect, Trino reads it, then it is dropped). Needs `spark` and `trino`. |
| tests.resources | object | `{"limits":{"memory":"1Gi"},"requests":{"cpu":"100m","memory":"256Mi"}}` | Test pod resources. |

