# Kubernetes / Helm Deployment Plan

Status: **DRAFT, reviewed** (2026-10-10, review fixes applied: §0, seaweedfs headless Service, Spark Connect static conf/`.master()` conflict, phase order, offline lint, Gateway caveats). Nothing in `helm/` exists yet. This document is the plan to build it.

Goal: run the same platform that `docker-compose.yaml` runs (same services, same images, same versions) on
Kubernetes as a Helm chart in a new `helm/` directory. The chart targets a local **minikube** cluster with an
NVIDIA GPU, and it uses current Kubernetes features where they bring a real benefit.

---

## 0. Rules for the implementing agent (read first)

- **Do not change any version or image** listed in §3. If something does not work at a pinned version, stop and report. Do not upgrade or downgrade to make it pass.
- **Compose must keep working.** Every shared file you touch (`docker-compose.yaml`, Dockerfiles, `trino/etc`, notebooks, `spark-defaults.conf`)
  must still pass `tests/e2e/run-e2e.sh --build` (19/19).
- **Never write `RELEASE_NOTES.md`** without asking the user (CLAUDE.md).
- Commit after each phase in §13, with that phase's "done when" evidence (command and output) in the commit message body.
- **Where each check can run.** A cloud agent container typically has no GPU and may have no Docker daemon. This one had 4 CPUs, 15 GiB RAM and no daemon.
  - Static checks (§12.1) run anywhere.
  - The minikube e2e (§12.2) needs a Docker daemon, and roughly 12 GiB for the lakehouse core, because Trino alone has `-Xmx8G`.
  - The GPU, DRA and notebook checks (§12.3–4) need the user's GPU host.
  - If a check can't run where you are, say so explicitly and hand it to the user. Never mark it as passed.
- When reality contradicts this plan (a field name, a default, a command), follow the upstream source, note the difference in the
  phase's commit message, and fix this document in the same commit.

---

## 1. Decisions taken (from the review Q&A)

| Topic | Decision |
|---|---|
| Spark runtime | **Kubeflow Spark Operator** (chart 2.5.2) replaces the standalone master/worker. |
| Notebook ↔ Spark | **Chart-managed `SparkConnect` CR** running *our* `datalab-playground/spark` image (Spark 4.1.3). Jupyter connects with pyspark 4.1.3 Spark Connect, either through `SPARK_REMOTE=sc://…` or the **Kubeflow SDK 0.5.0 in `connect(base_url=…)` mode** (installed without the `[spark]` extra). Batch jobs and the e2e test run as chart-templated `SparkApplication`s. |
| GPU for Ollama | **Device plugin by default** (`nvidia.com/gpu`, the path the minikube docs cover). An **optional DRA mode** (`ResourceClaimTemplate`, NVIDIA DRA driver) sits behind a values flag. |
| Image tags | **Pin everything.** `latest` is resolved to concrete versions in *both* compose and Helm (§3). |
| Host access | **Gateway API** (`Gateway` + `HTTPRoute`), implemented by **Envoy Gateway v1.9.2**. The setup script installs Envoy Gateway, and `minikube tunnel` exposes it. |
| Plan location | `docs/K8S_HELM_PLAN.md` (this file). |

### Why chart-managed SparkConnect rather than Kubeflow SDK "create" sessions (verified in source)

The Kubeflow SDK 0.5.0 (`kubeflow.spark`) can create a SparkConnect per notebook with `SparkClient().connect()`, but:
- `kubeflow/spark/backends/kubernetes/constants.py` hardcodes `DEFAULT_SPARK_VERSION = "4.0.4"` and `DEFAULT_SPARK_IMAGE = "apache/spark:4.0.4"`.
  `build_spark_connect_cr()` receives no Spark version from `SparkClient.connect()`, so it always uses 4.0.4.
- In create mode it also injects `spark.jars=https://repo1.maven.org/.../spark-connect_2.13-4.0.4.jar`, which is a runtime download from Maven.
- The `kubeflow[spark]` extra pins `pyspark-connect==4.2.0`.

Any of these would break the 4.1.3 / Iceberg 1.12.0 parity rule (CLAUDE.md pitfall 8). Instead, the chart deploys one
long-lived `SparkConnect` (`sparkoperator.k8s.io/v1alpha1`), and the operator runs it **inside our image**: the controller
executes `${SPARK_HOME}/sbin/start-connect-server.sh` (spark-operator `internal/controller/sparkconnect/options.go`), so
the server's Spark version is the one in our image (4.1.3), not the operator's.

### Kubeflow SDK: what is used and what is blocked (verified 2026-10-10)

The SDK **is installed** in the Jupyter image, but only the parts that don't depend on a Spark version are used:

| SDK 0.5.0 feature | Status | Reason (source) |
|---|---|---|
| `SparkClient().connect(base_url="sc://datalab-spark-server:15002")` | ✅ **used** | It only calls `SparkSession.builder.remote(base_url)` (`kubeflow/spark/api/spark_client.py`), so it works with any server version |
| `list_sessions()`, `get_session()`, `get_session_logs()` | ✅ **used** | Read-only on SparkConnect CRs and pods. Needs the Jupyter RBAC in §4. |
| `connect()` **create mode** (a SparkConnect per notebook) | ⛔ **blocked** | `spark_version` is always `DEFAULT_SPARK_VERSION = "4.0.4"` (no parameter on `connect()`). It also injects `spark-connect_2.13-4.0.4.jar` from Maven at runtime, which would clash with our 4.1.3 image. |
| `submit_job()` (FileJob / FuncJob) | ⛔ **blocked** | `build_spark_application_cr` hardcodes `sparkVersion` and `image: apache/spark:4.0.4`. That image lacks our hadoop-aws/Iceberg jars. |
| `pip install kubeflow[spark]` | ⛔ **blocked** | The extra pins `pyspark-connect==4.2.0`. pip reports `ResolutionImpossible` next to `pyspark==4.1.3` (tested). |

Install line (tested: resolves, and `from kubeflow.spark import SparkClient` imports with pyspark 4.1.3):
`pip install 'pyspark[connect]==4.1.3' 'kubeflow==0.5.0' 'kubeflow-spark-api>=2.4.0,<2.5'`. This resolved to `kubeflow-spark-api 2.4.0`. Pin the exact versions you get.

Optional, unverified (Phase 3, don't rely on it): `submit_job(..., options=[PodTemplateOverride(...)])` might be able to override the container image.
If it works with our image, document it. If not, record it as blocked too.

**On every SDK or operator upgrade**, re-check the blocked rows (the commands are also in `docs/VERSIONS.md` §6):
```bash
pip download --no-deps kubeflow==<new> -d /tmp/kf && unzip -o -q /tmp/kf/kubeflow-*.whl -d /tmp/kf/x
grep -n "DEFAULT_SPARK_VERSION\|DEFAULT_SPARK_IMAGE" /tmp/kf/x/kubeflow/spark/backends/kubernetes/constants.py
grep -n "spark_version\|image" /tmp/kf/x/kubeflow/spark/api/spark_client.py   # look for new connect()/submit_job() parameters
grep -n "pyspark" /tmp/kf/x/kubeflow-*.dist-info/METADATA                      # the [spark] extra's pyspark-connect pin
```
Unblocked when: `connect()`/`submit_job()` accept a Spark version **and** an image, and the `[spark]` extra allows `pyspark==4.1.x`
(or the default Spark version equals ours). Then enable create mode and SDK job submission and drop the chart's fixed SparkConnect if it's no longer needed.

---

## 2. Verified upstream facts and sources

| Item | Value used | Source (checked 2026-10-10) |
|---|---|---|
| minikube | **v1.39.0** (2026-09-01), default Kubernetes **v1.37.0**, supports 1.36 and 1.37 | github.com/kubernetes/minikube/releases |
| minikube `nvidia-device-plugin` addon | v0.20.0 (bumped in minikube 1.39.0) | same release notes |
| Kubernetes | latest stable **v1.37.1**. v1.37 "Garhwal" was released 2026-08-26. | github.com/kubernetes/kubernetes/releases, kubernetes.io/blog/2026/08/26/kubernetes-v1-37-release/ |
| Helm | **v4.3.0** | github.com/helm/helm tags |
| Kubeflow Spark Operator | chart/app **2.5.2**, repo `https://kubeflow.github.io/spark-operator`. Its image is based on `apache/spark:4.0.4`. | kubeflow/spark-operator@v2.5.2 `charts/spark-operator-chart/Chart.yaml`, `Dockerfile` |
| SparkConnect server Service name | `<sparkconnect-name>-server` (port 15002) | spark-operator `internal/controller/sparkconnect/util.go` |
| Envoy Gateway | **v1.9.2**, `helm install eg oci://docker.io/envoyproxy/gateway-helm --version v1.9.2 -n envoy-gateway-system --create-namespace`. The chart also installs the Gateway API CRDs (`crds.enabled`). | envoyproxy/gateway@v1.9.2 docs and `charts/gateway-helm/README.md` |
| Gateway API | v1.6.3 (latest tag) | kubernetes-sigs/gateway-api tags |
| NVIDIA DRA driver | **v0.5.0**, `oci://registry.k8s.io/dra-driver-nvidia/charts/dra-driver-nvidia-gpu`, `kubeVersion >=1.32`, DeviceClass **`gpu.nvidia.com`** | NVIDIA/k8s-dra-driver-gpu@v0.5.0 releases, chart, demo specs |
| Getting local images into minikube | `eval $(minikube docker-env)` + `docker build` needs `--container-runtime=docker`. `minikube image load` works on every runtime. `imagePullPolicy` must not be `Always`. | minikube `site/content/en/docs/handbook/pushing.md` |
| minikube + NVIDIA | the docker driver: `nvidia-smi` works on the host, `net.core.bpf_jit_harden=0`, NVIDIA Container Toolkit, `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`, then `minikube start --driver docker --container-runtime docker --gpus all` | minikube `site/content/en/docs/tutorials/nvidia.md` |

---

## 3. Image and version parity

Every image is identical between compose and Helm. A single table in `docs/VERSIONS.md` stays authoritative.

| Service | Image (compose == Helm) | Change |
|---|---|---|
| seaweedfs | `chrislusf/seaweedfs:4.48` | — |
| metastore-db | `postgres:13` | — |
| hive-metastore | `datalab-playground/hive-metastore:latest` (local build, HMS 4.2.1) | — |
| trino | `datalab-playground/trino:latest` (local build, Trino 483) | — |
| spark (SparkConnect server, executors, SparkApplications) | `datalab-playground/spark:latest` (local build, Spark 4.1.3) | Possibly add `spark-connect_2.13-4.1.3.jar` at build time; see Phase 1. |
| jupyter | `datalab-playground/jupyter:latest` (local build) | `pyspark[connect]==4.1.3` instead of `pyspark==4.1.3`, so the Connect client deps (grpcio etc.) are present. Add `kubeflow==0.5.0` + `kubeflow-spark-api` (pinned, **no `[spark]` extra**, §1). |
| phoenix | `arizephoenix/phoenix:latest` → **`arizephoenix/phoenix:version-20.20.0`** | pin (same digest as `latest` today: `sha256:3a2e5a04…`) |
| phoenix db | `postgres:17` | — |
| ollama | `ollama/ollama:latest` → **`ollama/ollama:0.40.2`** | pin (same digest as `latest` today: `sha256:b86366bb…`) |
| qdrant | `qdrant/qdrant:latest` → **`qdrant/qdrant:v1.19.2`** | pin (same digest as `latest` today: `sha256:b7b0444c…`) |

Rules:
- The local images keep the compose names (`datalab-playground/<svc>:latest`). The chart sets `imagePullPolicy: IfNotPresent` explicitly,
  because Kubernetes defaults `:latest` to `Always`, which would try Docker Hub and fail.
- The chart's `values.yaml` holds the third-party tags, and a CI-able script (`helm/scripts/check-parity.sh`) diffs them against
  `docker-compose.yaml` so the two can't drift.
- The same script also diffs every key in `spark/conf/spark-defaults.conf` (except `spark.master`) against the chart's `spark.sparkConf`
  values. This adds a third copy to keep in sync, on top of CLAUDE.md pitfall 2.
- Not added to the chart: the bitnami or other upstream Postgres/Trino/Qdrant charts. Those ship different images, which would break parity.

---

## 4. Architecture on Kubernetes

Key principle: **the Service names equal the compose service names** (`seaweedfs`, `metastore-db`, `hive-metastore`, `trino`,
`phoenix`, `db`, `ollama`, `qdrant`, `jupyter`). Every config file baked into the images keeps working unchanged:
`trino/etc/catalog/*.properties`, `hive-metastore/*.xml.template`, `spark-defaults.conf`, and the notebooks' `http://ollama:11434`
and `phoenix:4317`. Everything goes in one namespace, `datalab`.

```
                        minikube tunnel → Envoy Gateway (Gateway "datalab", HTTP :80)
                          │ HTTPRoutes by hostname: jupyter / trino / phoenix / ollama / qdrant / s3 / filer / spark-ui
 ┌──────────────────────── namespace: datalab ─────────────────────────────────────────────┐
 │ jupyter (Deployment) ── SPARK_REMOTE=sc://datalab-spark-server:15002 ─▶ SparkConnect CR │
 │   │                                       (operator → server pod + executor pods,       │
 │   │                                        image datalab-playground/spark, Spark 4.1.3) │
 │   ├─▶ trino (Deployment) ─ hive/lakehouse ─▶ hive-metastore :9083 ┐                     │
 │   │                       └ iceberg ───────▶ hive-metastore :9084 ┴▶ metastore-db (STS) │
 │   │   SparkConnect iceberg_catalog ────────▶ hive-metastore :9084/iceberg               │
 │   │   all S3 traffic ──────────────────────▶ seaweedfs (STS) :8333  s3a://warehouse/    │
 │   ├─▶ ollama (Deployment, GPU) ◀── ollama-pull (Job)                                    │
 │   ├─▶ qdrant (STS)   └─▶ phoenix (Deployment) ─▶ db (STS, postgres:17)                  │
 │ seaweedfs-bucket (Job): weed shell s3.bucket.create -name warehouse                     │
 └─────────────────────────────────────────────────────────────────────────────────────────┘
 namespace: spark-operator      → Kubeflow Spark Operator 2.5.2 (jobNamespaces: [datalab])
 namespace: envoy-gateway-system → Envoy Gateway v1.9.2
 (optional) namespace: dra-driver-nvidia-gpu → NVIDIA DRA driver v0.5.0
```

### Compose → Kubernetes mapping

| Compose service | Kubernetes object(s) | Notes |
|---|---|---|
| seaweedfs | StatefulSet (1) + PVC `/data` + **headless** Service `seaweedfs` (`clusterIP: None`, `publishNotReadyAddresses: true`) + Secret `s3.json` | The same `server … -volume.max=64 …` args. The master, volume, filer and S3 processes in the pod reach each other at `-ip=seaweedfs`. Those connections use the HTTP ports **and** their gRPC ports (+10000) and the volume port 8080. With a normal ClusterIP Service, every port would need listing, and the pod would hairpin back to itself through its own Service. A headless Service makes `seaweedfs` resolve to the pod IP, exactly like compose. `publishNotReadyAddresses` is needed because the processes must resolve the name before the pod is Ready. Readiness is `httpGet /healthz :8333` (compose healthcheck). |
| (start-platform bucket step) | Job `seaweedfs-bucket` | Idempotent `weed shell -master=seaweedfs:9333` create-or-list, the same as `start-platform.sh`. `backoffLimit` plus `ttlSecondsAfterFinished`. |
| metastore-db | StatefulSet + PVC + Service `metastore-db:5432` + Secret | `pg_isready` exec readiness probe (compose healthcheck) |
| hive-metastore | Deployment (1, `strategy: Recreate`) + Service (9083, 9084) | An initContainer waits for `metastore-db` (`pg_isready`) and `seaweedfs` (`/healthz`), which replaces `depends_on: service_healthy`. The startupProbe is TCP 9083 with a long budget for the schema init. Readiness also checks `GET :9084/iceberg/v1/config`. Same `SERVICE_NAME`, `DB_DRIVER`, `SERVICE_OPTS` env. |
| trino | Deployment + Service 8080 | Uses the config baked into the image (compose additionally bind-mounts `./trino/etc`). An optional `values.trino.configOverride` renders a ConfigMap. Readiness is `/v1/info` `"starting":false`. Memory request/limit sized for `-Xmx8G` in `jvm.config` (see §11). |
| spark-master, spark-worker | **removed**. Replaced by the SparkConnect CR `datalab-spark` plus the Spark Operator. | The executors are pods. `spark-defaults.conf` keys move to `sparkConf`/`hadoopConf` rendered from one values block (§5). |
| jupyter | Deployment + Service 8888 + PVC or hostPath for notebooks + ServiceAccount `jupyter` + Role/RoleBinding | Env `SPARK_REMOTE`, `AWS_REGION`. The notebooks come from `minikube mount` → hostPath (the default for minikube), or from a PVC. The Role allows the SDK's read-only calls: `get/list/watch` on `sparkoperator.k8s.io` `sparkconnects`, plus `pods` and `pods/log`. The `create`/`delete` verbs stay out until create mode is unblocked (§1). |
| phoenix | Deployment + Service (6006, 4317, 9090) | An initContainer waits for `db`. `PHOENIX_SQL_DATABASE_URL` comes from a Secret. |
| db (phoenix-db) | StatefulSet + PVC + Service **`db`**:5432 | Keeps the name `db`, because Phoenix's URL uses it |
| ollama | Deployment (`Recreate`) + PVC `/root/.ollama` + Service 11434 | GPU via `nvidia.com/gpu: 1` or a DRA claim (§6). Readiness is `exec ollama list` (compose healthcheck). |
| ollama-init | Job `ollama-pull` | The ollama image with `OLLAMA_HOST=http://ollama:11434` runs `ollama pull gemma3:4b` and `ollama pull mxbai-embed-large:latest`. Models are listed in values. It needs no GPU, because the server does the download. |
| qdrant | StatefulSet + PVC + Service (6333, 6334) | The same `QDRANT__SERVICE__*` env |

---

## 5. Spark with the Kubeflow Spark Operator

1. **Operator**: the `helm/scripts/setup-minikube.sh` script installs it, kept separate from our chart so the CRDs have their own lifecycle:
   `helm repo add spark-operator https://kubeflow.github.io/spark-operator` then
   `helm install spark-operator spark-operator/spark-operator --version 2.5.2 -n spark-operator --create-namespace --set "spark.jobNamespaces={datalab}"`.
   The operator chart creates the `spark-operator-spark` ServiceAccount and RBAC in each job namespace (values `spark.jobNamespaces`, `spark.serviceAccount`).
   Our chart declares a hard dependency only through a `helm template`-time check (`.Capabilities.APIVersions.Has "sparkoperator.k8s.io/v1alpha1/SparkConnect"`), which fails with a clear message.
2. **SparkConnect CR `datalab-spark`** (templated). Its Service is `datalab-spark-server:15002`.
   - `sparkVersion: 4.1.3`, `image: datalab-playground/spark:latest`, `imagePullPolicy: IfNotPresent` in the server and executor templates.
   - `sparkConf` comes from one values block that mirrors `spark/conf/spark-defaults.conf`: the S3A endpoint and keys, the timeouts, Kryo, AQE, and the pyspark python path. It also adds the Iceberg REST catalog that the e2e test and notebook set at runtime: `spark.sql.catalog.iceberg_catalog=org.apache.iceberg.spark.SparkCatalog`, `.type=rest`, `.uri=http://hive-metastore:9084/iceberg`, `.io-impl=org.apache.iceberg.hadoop.HadoopFileIO`.
     It also needs **`spark.sql.extensions=org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions`**. This is a *static* conf: under Spark Connect, a client-side
     `.config("spark.sql.extensions", …)` is silently dropped (pyspark 4.1.3 `sql/connect/session.py` `_apply_options` swallows errors for static confs), so the
     Iceberg `CALL`/DDL extensions would be missing. Add the extra `spark.hadoop.fs.s3a.*` keys the notebook sets per session too (magic committer, fast upload, multipart sizes),
     so behaviour doesn't depend on runtime-conf propagation.
     `spark.master` is **not** set, because the operator owns it.
   - Env `AWS_REGION=us-east-1` on the server and executor templates (and on SparkApplication driver/executor), as compose sets it on every Spark container.
   - Executors: `instances: 1`, `cores: 2`, `memory: 2g`, which equals the compose worker (2 cores / 2g). Optional `dynamicAllocation` uses the CRD fields `minExecutors`/`maxExecutors`/`shuffleTrackingEnabled`, off by default.
   - Pod securityContext follows the operator example: `runAsUser/runAsGroup 185`, `runAsNonRoot`, `drop: [ALL]`, `seccompProfile: RuntimeDefault`.
3. **Jupyter** gets `SPARK_REMOTE=sc://datalab-spark-server:15002`. With that variable set, pyspark's `SparkSession.builder…getOrCreate()` returns a Spark Connect session.
   **The notebooks still need changes.** These were verified in the pyspark 4.1.3 source and the notebook:
   - `data_lab_playground.ipynb` cells 8, 18 and 20 call `.master("spark://spark-master:7077")`. In pyspark 4.1.3, `_validate_startup_urls` raises
     `CANNOT_CONFIGURE_SPARK_CONNECT_MASTER` when `spark.master` is set while `SPARK_REMOTE` is set.
     **Fix:** remove the `.master(...)` calls. On compose the master still comes from `spark.master=spark://spark-master:7077` in the Jupyter image's `spark-defaults.conf`,
     so compose behaviour is unchanged. Prove this in Phase 1 with the compose run.
   - Cells 8 and 20 use `spark.sparkContext.master` and `.sparkContext.getConf()`. SparkContext does not exist under Spark Connect.
     Guard them with `if not os.environ.get("SPARK_REMOTE")` and print `spark.conf.get(...)` values otherwise.
   - Also grep `rag_demo.ipynb` and `trino_query_example_updated.py` for `.master(`, `sparkContext` and `.rdd`.
   - Add one short "Spark on Kubernetes" cell to `data_lab_playground.ipynb`, skipped on compose (no `SPARK_REMOTE`). It shows
     `SparkClient().connect(base_url=os.environ["SPARK_REMOTE"])`, `list_sessions()` and `get_session_logs("datalab-spark")`.
4. **Batch / e2e**: a `SparkApplication` template (off by default; the e2e enables it) runs `tests/e2e/e2e_lakehouse.py` from a ConfigMap with the same image and `sparkConf`.
   The script is mounted through `spec.volumes` and `driver.volumeMounts`. The operator's mutating webhook does that mounting, so keep `webhook.enable=true`, which is the chart default.
   Caveat: the operator submits with its own `spark-submit` (Spark 4.0.4) against our 4.1.3 image. Phase 2 must prove this works, and §12 lists the fallback.
5. **Spark UI**: the SparkConnect server is the driver, so its UI is on 4040. The plan adds an `HTTPRoute` to it if the operator's server Service exposes 4040. If it doesn't, the chart adds a small extra Service selecting the server pod (verify in Phase 2).

---

## 6. GPU for Ollama

**Default: device plugin.** `minikube start --gpus all` (docker driver) plus the `nvidia-device-plugin` addon advertise `nvidia.com/gpu`.
The Ollama container requests `resources.limits: {nvidia.com/gpu: 1}`. The compose-only settings (`runtime: nvidia`, `NVIDIA_VISIBLE_DEVICES`) are not
carried over, because the device plugin injects the GPU.

**Optional: DRA** (`gpu.mode: dra` in values). It is stable Kubernetes, and v1.37 graduated more DRA features (per the release blog).
- The chart renders a `ResourceClaimTemplate` (`resource.k8s.io/v1`) with `deviceClassName: gpu.nvidia.com`. The Ollama pod gets `spec.resourceClaims`, and the container gets `resources.claims`. The chart guards this with `.Capabilities.APIVersions.Has "resource.k8s.io/v1"`.
- Prerequisite, done by the setup script with `--gpu-mode dra`: `helm install dra-driver-nvidia-gpu oci://registry.k8s.io/dra-driver-nvidia/charts/dra-driver-nvidia-gpu --version 0.5.0 -n dra-driver-nvidia-gpu --create-namespace --set gpuResourcesEnabledOverride=true`.
- Neither the minikube docs nor the DRA driver release notes say whether the device plugin and the DRA driver can share one GPU.
  The script therefore runs `minikube addons disable nvidia-device-plugin` in DRA mode. The DRA path is labelled **experimental** until validated on real hardware.

---

## 7. Host access with the Gateway API

- `setup-minikube.sh` installs Envoy Gateway v1.9.2 (this also installs the Gateway API CRDs).
- Our chart renders a `GatewayClass` `datalab` (controller `gateway.envoyproxy.io/gatewayclass-controller`; `gateway.createClass: false` to reuse an existing class) and a `Gateway` `datalab` with an HTTP listener on :80.
  It also renders one `HTTPRoute` per UI/API, using hostnames under `values.gateway.domain` (default `datalab.test`):

| Hostname | Backend | Compose equivalent |
|---|---|---|
| `jupyter.datalab.test` | jupyter:8888 | localhost:8888 |
| `trino.datalab.test` | trino:8080 | localhost:8080 |
| `phoenix.datalab.test` | phoenix:6006 | localhost:6006 |
| `ollama.datalab.test` | ollama:11434 | localhost:11434 |
| `qdrant.datalab.test` | qdrant:6333 | localhost:6333 |
| `s3.datalab.test` | seaweedfs:8333 | localhost:8333 |
| `filer.datalab.test` / `seaweed.datalab.test` | seaweedfs:8888 / 9333 | localhost:8889 / 9333 |
| `spark.datalab.test` | SparkConnect driver UI :4040 | localhost:8081 (master UI) |

- **Trino behind a proxy:** Envoy adds `X-Forwarded-*` headers. By default, Trino rejects requests that carry them unless `http-server.process-forwarded=true` is set.
  Verify this in Phase 4 by loading `trino.datalab.test/ui/`. If it is rejected, add that line to `trino/etc/config.properties`. It is harmless for compose, where no proxy sends the headers.
- **Jupyter kernels use WebSockets.** Phase 4 must open a notebook and run a cell through `jupyter.datalab.test`, not just load the page.
  If the upgrade fails, check Envoy Gateway's upgrade/WebSocket settings (a `ClientTrafficPolicy` or `BackendTrafficPolicy`) in the v1.9 docs.
- **Not exposed through the Gateway**: Postgres x2, HMS Thrift/REST, Qdrant gRPC, Phoenix OTLP gRPC, and Spark Connect gRPC. These are in-cluster only, as they are in compose for clients.
  `helm/scripts/port-forward.sh` provides `kubectl port-forward` for the occasional host access (for example DBeaver to :5433, or a local pyspark to 15002).
- Run `minikube tunnel` in a separate terminal to give the Envoy LoadBalancer Service an address. `helm/scripts/hosts.sh` reads
  `kubectl get gateway datalab -o jsonpath='{.status.addresses[0].value}'` and prints the `/etc/hosts` line for the hostnames above. It edits nothing itself.

---

## 8. Kubernetes features used (all GA in ≤1.37 unless marked)

- `Chart.yaml` `apiVersion: v2`, `kubeVersion: ">=1.36.0-0"`, matching minikube 1.39's supported range. Ships `values.schema.json` so bad values fail at `helm install`.
- **StatefulSets** with `volumeClaimTemplates` and `persistentVolumeClaimRetentionPolicy` (`whenDeleted: Retain` by default) for the data stores.
- **startup / readiness / liveness probes**: the compose healthchecks are translated one to one, and the startupProbes replace the `sleep`s in `start-platform.sh`.
- **initContainers** for dependency waits, in place of `depends_on: condition: service_healthy`.
- **Jobs** with `backoffLimit`, `ttlSecondsAfterFinished` and `podFailurePolicy` for the bucket and model-pull steps.
- **Gateway API** v1 `Gateway`/`HTTPRoute` (§7).
- **DRA** `resource.k8s.io/v1` `ResourceClaimTemplate` (optional, §6).
- **Pod Security Admission** namespace labels (`pod-security.kubernetes.io/warn: baseline`) so violations are visible without blocking the local dev images that run as root.
- **Secrets** for every credential (the values stay the same hardcoded dev values, but no longer sit inline in env).
- **Recommended labels** `app.kubernetes.io/*` everywhere, so `kubectl get all -l app.kubernetes.io/part-of=datalab` works.
- The `.Capabilities` checks fail fast when the Spark Operator, Gateway API or DRA CRDs/APIs are missing.
  Offline, `helm lint` and `helm template` see no CRDs, so the static tests must pass
  `--api-versions sparkoperator.k8s.io/v1alpha1/SparkConnect --api-versions sparkoperator.k8s.io/v1beta2/SparkApplication --api-versions gateway.networking.k8s.io/v1/Gateway --api-versions resource.k8s.io/v1/ResourceClaimTemplate`.
  Put that in a small `helm/scripts/lint.sh`, so nobody "fixes" the checks by deleting them.
- **PSA `baseline` warnings are expected** for the `hostPath` notebook mount and for root containers. Don't silence them by changing images.

---

## 9. Directory layout

```
helm/
├── README.md                      # quick start, mirrors §10 and §11
├── datalab/                       # the umbrella chart (single chart, no subcharts)
│   ├── Chart.yaml
│   ├── values.yaml                # images and tags (== compose), resources, gpu.mode, gateway, spark conf
│   ├── values.schema.json
│   ├── values-minikube.yaml       # hostPath notebook mount, smaller resources
│   ├── files/
│   │   └── e2e_lakehouse.py       # symlink/copy of tests/e2e/e2e_lakehouse.py (for the SparkApplication ConfigMap)
│   └── templates/
│       ├── _helpers.tpl  NOTES.txt  namespace-psa.yaml  secrets.yaml
│       ├── seaweedfs/     statefulset, service, configmap(s3.json), job-bucket
│       ├── metastore-db/  statefulset, service
│       ├── hive-metastore/deployment, service
│       ├── trino/         deployment, service, configmap (optional)
│       ├── spark/         sparkconnect.yaml, sparkapplication-e2e.yaml, service-ui.yaml
│       ├── jupyter/       deployment, service, pvc
│       ├── phoenix/       deployment, service, db-statefulset, db-service
│       ├── ollama/        deployment, service, pvc, job-pull, resourceclaimtemplate (dra)
│       ├── qdrant/        statefulset, service
│       └── gateway/       gatewayclass, gateway, httproutes
└── scripts/
    ├── setup-minikube.sh          # §10: preflight + minikube start + operator + Envoy (+ DRA)
    ├── build-images.sh            # §11: build the 4 local images into minikube
    ├── deploy.sh                  # helm upgrade --install datalab ./helm/datalab -n datalab --create-namespace --wait
    ├── hosts.sh  port-forward.sh  check-parity.sh
    └── e2e-k8s.sh                 # §12
```

---

## 10. Local minikube cluster with GPU (from the minikube NVIDIA tutorial)

`helm/scripts/setup-minikube.sh` automates the steps below, and `helm/README.md` documents them by hand.
Each step follows `site/content/en/docs/tutorials/nvidia.md` (docker driver path). Linux only, as the tutorial states.
macOS and Windows are not supported for NVIDIA GPUs.

```bash
# 0. Host prerequisites (one time)
nvidia-smi                                         # an NVIDIA driver must be installed
sudo sysctl net.core.bpf_jit_harden                # must print 0; otherwise:
echo "net.core.bpf_jit_harden=0" | sudo tee -a /etc/sysctl.conf && sudo sysctl -p
# install the NVIDIA Container Toolkit (NVIDIA's install guide), then:
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
# install minikube v1.39.0, kubectl (v1.37.x) and helm v4

# 1. Cluster (if minikube existed before the NVIDIA runtime: `minikube delete` first)
minikube start --driver docker --container-runtime docker --gpus all \
  --kubernetes-version=v1.37.0 --cpus <N> --memory <M> --disk-size <D> \
  --mount --mount-string="$PWD/jupyter/notebooks:/datalab/notebooks"
#   CDI alternative from the same doc: --gpus nvidia.com

# 2. Check the GPU is advertised (verification command from the minikube docs)
kubectl get nodes -ojson | jq '.items[].status.capacity'   # expect "nvidia.com/gpu"
#   if it is missing: minikube addons enable nvidia-device-plugin

# 3. Platform prerequisites
helm repo add spark-operator https://kubeflow.github.io/spark-operator && helm repo update
helm install spark-operator spark-operator/spark-operator --version 2.5.2 \
  -n spark-operator --create-namespace --set "spark.jobNamespaces={datalab}"
helm install eg oci://docker.io/envoyproxy/gateway-helm --version v1.9.2 \
  -n envoy-gateway-system --create-namespace
# optional, --gpu-mode dra only:
#   minikube addons disable nvidia-device-plugin
#   helm install dra-driver-nvidia-gpu oci://registry.k8s.io/dra-driver-nvidia/charts/dra-driver-nvidia-gpu \
#     --version 0.5.0 -n dra-driver-nvidia-gpu --create-namespace --set gpuResourcesEnabledOverride=true
```

`--container-runtime docker` is required by the GPU docs, and it is also what enables the `docker-env` image path (§11).
The script has flags: `--cpus/--memory/--disk-size`, `--gpu-mode device-plugin|dra|none`. `none` exists for CI or machines without a GPU, and it deploys Ollama
without a GPU request, so it runs on CPU, much like the compose `run-e2e.sh`, which skips the GPU services.

---

## 11. Building and loading images: `helm/scripts/build-images.sh`

```bash
eval $(minikube docker-env)                       # the build lands directly in the node's Docker
for svc in hive-metastore trino spark jupyter; do
  docker build -t "datalab-playground/$svc:latest" "./$svc"
done
# pre-pull the pinned third-party images into the node so the first deploy doesn't stall
docker pull chrislusf/seaweedfs:4.48 postgres:13 postgres:17 \
  arizephoenix/phoenix:version-20.20.0 ollama/ollama:0.40.2 qdrant/qdrant:v1.19.2   # one per line in the script
```

- Same Dockerfiles, same tags as `start-platform.sh`. The script reuses its "rebuild only if sources changed" logic, comparing against `docker image inspect` inside minikube.
- Fallback flag `--load` (any runtime): build on the host and then run `minikube image load datalab-playground/<svc>:latest`.
- The third-party image list is read from `helm/datalab/values.yaml`, so there is one source.

Resource sizing (to confirm in Phase 2): Trino's `jvm.config` has `-Xmx8G`, so the Trino container needs a memory limit above 8 GiB.
Add the Spark Connect server, one 2g executor, Ollama with a 4B model, Jupyter and the rest. The proposed minikube default is
`--cpus 8 --memory 24g --disk-size 80g`. `values-minikube.yaml` keeps requests well below that.

---

## 12. Testing

1. **Static** (`helm/scripts/lint.sh`): `helm lint helm/datalab`, plus `helm template` for each of `gpu.mode=device-plugin|dra|none` (with the `--api-versions` flags from §8). Validate the output with `kubeconform` against the 1.37 schemas and the Gateway API, SparkConnect and DRA CRD schemas.
   Also run `check-parity.sh`, which fails if any image tag differs from compose.
2. **e2e on Kubernetes** (`helm/scripts/e2e-k8s.sh`, no GPU needed, `--gpu-mode none`): deploy only the lakehouse core
   (`seaweedfs, metastore-db, hive-metastore, trino, spark`) with `--set` toggles, wait on readiness, then run `e2e_lakehouse.py` as a
   `SparkApplication` against the same 19 checks as `tests/e2e/run-e2e.sh`.
   - **Fallback** if the operator's Spark 4.0.4 `spark-submit` can't submit our 4.1.3 image: run the same script as a plain `Job` using our spark image
     in `--master k8s://https://kubernetes.default.svc` client mode with a ServiceAccount. Record the outcome in `docs/VERSIONS.md`.
   - Also run the e2e's Spark half through Spark Connect (`SPARK_REMOTE`), which is the path notebooks use.
3. **GPU smoke** (manual, GPU host): run `ollama-pull` to completion, then call `POST /api/generate` through `ollama.datalab.test` and confirm `nvidia-smi` in the ollama pod shows the process. Repeat in DRA mode.
4. **Notebooks**: run `data_lab_playground.ipynb` and `rag_demo.ipynb` end to end on Kubernetes, and run them once more on compose to prove the `SPARK_REMOTE` guards didn't break compose.

---

## 13. Implementation phases

| Phase | Work | Done when |
|---|---|---|
| 1. Parity prep | Pin phoenix/ollama/qdrant in `docker-compose.yaml`. Jupyter: `pyspark[connect]==4.1.3`. Check `ls /opt/spark/jars \| grep connect` in `apache/spark:4.1.3`. It is expected to be present, because Spark's `assembly/pom.xml` depends on `spark-connect_2.13`. Only if `spark-connect_2.13-4.1.3.jar` is missing, add it in `spark/Dockerfile` at build time (no runtime Maven download). Notebook fixes from §5.3. | compose `run-e2e.sh --build` 19/19 still passes, and notebook cells 8/18/20 run on compose |
| 1b. Cluster tooling | `setup-minikube.sh` (`--gpu-mode none` first), `build-images.sh`, `deploy.sh`, `lint.sh`, `check-parity.sh`. These are needed by every later phase. | a minikube with the Spark Operator and Envoy Gateway running, and the four local images visible in `minikube image ls` |
| 2. Chart: lakehouse core | seaweedfs (+ bucket Job), metastore-db, hive-metastore, trino, SparkConnect, SparkApplication-e2e. Secrets, probes, PSA labels, schema, `e2e-k8s.sh`. | `lint.sh` clean, and `e2e-k8s.sh` 19/19 on minikube with `--gpu-mode none`. Also the Spark-Connect variant of the e2e. |
| 3. Chart: AI + Jupyter | jupyter (+ SDK ServiceAccount/Role), phoenix + db, qdrant, ollama (device plugin) + pull Job. Try the optional `PodTemplateOverride` image test (§1) and record the result. | the SDK cell works on minikube (`connect(base_url=…)`, `list_sessions`, `get_session_logs`). On the user's GPU host: both notebooks run end to end, and `nvidia-smi` in the ollama pod shows the model. |
| 4. Gateway + access | GatewayClass/Gateway/HTTPRoutes, `hosts.sh`, `port-forward.sh`, the Trino forwarded-header and Jupyter WebSocket checks from §7, and `helm/README.md` | a fresh-machine run from `helm/README.md` works, and every hostname in §7 answers |
| 5. DRA (experimental) | ResourceClaimTemplate + `--gpu-mode dra` path | Ollama gets the GPU through a ResourceClaim |
| 6. Docs | Post-change rule (CLAUDE.md): README, AGENTS.md, CLAUDE.md (new pitfalls: Service names == compose names, `IfNotPresent`, SparkConnect vs SparkContext, SDK 4.0.4 lock-in), `docs/VERSIONS.md` (Kubernetes/minikube/Helm/operator/Envoy/DRA/**Kubeflow SDK** rows and the image pins). In VERSIONS.md §6 "Upgrade blockers", add a row: *Kubeflow SDK create mode / `submit_job()` / `[spark]` extra — blocked by the hardcoded Spark 4.0.4 version and image and the `pyspark-connect==4.2.0` pin — re-check on every SDK release* with the commands from §1. Add a matching short pitfall to CLAUDE.md and AGENTS.md: "Kubeflow SDK: `base_url` mode only, never install `kubeflow[spark]`". RELEASE_NOTES only after you confirm. | docs reviewed |

---

## 14. Risks and open verification items

| # | Item | Mitigation |
|---|---|---|
| R1 | The operator's `spark-submit` 4.0.4 submits a Spark 4.1.3 app (SparkApplication only; SparkConnect runs `start-connect-server.sh` from our image) | The Phase 2 e2e proves it. Otherwise use the plain Job fallback (§12). |
| R2 | `apache/spark:4.1.3` might not ship the Spark Connect server jar (unlikely, because the assembly depends on it) | Phase 1 `ls` check. Add the jar at build time only if it is missing. |
| R3 | Spark Connect lacks SparkContext/RDD APIs | Notebook guards. Document the limitation. |
| R4 | SparkConnect CRD is `v1alpha1` (may change) | Pin operator 2.5.2. Re-check on upgrade (add to VERSIONS.md §6). |
| R4b | Kubeflow SDK's version-dependent features are locked to Spark 4.0.4 | `base_url` mode only. The VERSIONS.md §6 blocker row is re-checked on every SDK release (§1). |
| R5 | Device plugin + DRA on the same GPU is undocumented | DRA mode disables the device plugin addon. DRA stays experimental. |
| R6 | `minikube tunnel` address and `/etc/hosts` handling differ per host | `hosts.sh` prints the line rather than editing. `port-forward.sh` is the fallback. |
| R7 | Trino 8G heap on a small laptop | Minikube sizing in the script. An optional values override for `jvm.config` (it changes only when the user sets it, so the defaults keep parity). |
| R8 | HMS schema init runs on every pod start (same as compose, `IS_RESUME` unset) | Same behaviour as compose. Keep `Recreate` with a single replica. |
| R9 | The `minikube --mount` notebook folder may not be writable by Jupyter's uid 1000 | Verify in Phase 3 by saving a notebook. Fall back to a PVC (`jupyter.notebooks.mode: pvc`) plus `kubectl cp`. |
| R10 | Trino rejects `X-Forwarded-*` from Envoy | `http-server.process-forwarded=true` (§7) |
| R11 | Jupyter WebSockets through Envoy | Phase 4 runs a cell through the Gateway (§7) |
| R12 | Spark conf now lives in three places (two `spark-defaults.conf` files plus chart values) | `check-parity.sh` diff (§3). Update CLAUDE.md pitfall 2. |

---

## 15. Out of scope

- Production hardening (TLS, auth, HA, NetworkPolicies enforced by a CNI). The credentials stay the intentionally hardcoded dev values, now held in Secrets.
- Publishing the chart to a registry.
- Replacing compose. Both stay supported, with identical images.
