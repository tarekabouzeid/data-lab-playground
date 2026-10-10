# DataLab Playground on Kubernetes (Helm)

The same platform as `docker-compose.yaml`, as a Helm chart: **same images, same versions, same service names**.
Design and decisions: [docs/K8S_HELM_PLAN.md](../docs/K8S_HELM_PLAN.md). Versions: [docs/VERSIONS.md](../docs/VERSIONS.md).

| | Compose | Kubernetes |
|---|---|---|
| Object storage | `seaweedfs` | StatefulSet `seaweedfs` (+ Job that creates the `warehouse` bucket) |
| Catalog DB / HMS | `metastore-db`, `hive-metastore` | StatefulSet / Deployment, same names |
| SQL | `trino` | Deployment `trino` |
| Spark | `spark-master` + `spark-worker` | Kubeflow Spark Operator: one `SparkConnect` server (`datalab-spark-server:15002`), executors are pods |
| Notebooks | `jupyter` (classic Spark master) | Deployment `jupyter` (Spark Connect via `SPARK_REMOTE`) |
| AI | `ollama` (GPU), `qdrant`, `phoenix` + `db` | same names; Ollama gets the GPU from the device plugin (or DRA) |
| Host access | published ports | Gateway API (Envoy Gateway) hostnames, or `port-forward.sh` |

## Quick start (Linux + NVIDIA GPU)

Prerequisites: Docker, [minikube](https://minikube.sigs.k8s.io/docs/start/) v1.39, `kubectl`, [Helm](https://helm.sh/docs/intro/install/) v4, `jq`,
an NVIDIA driver, and for the GPU the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
helm/scripts/setup-minikube.sh                 # cluster + Spark Operator + Envoy Gateway   (--gpu-mode none for CPU only)
helm/scripts/build-images.sh                   # builds the 4 local images inside minikube + pulls the pinned third-party images
helm/scripts/deploy.sh                         # installs the chart (values.yaml + values-minikube.yaml)
minikube tunnel                                # separate terminal: gives the Gateway an address
helm/scripts/hosts.sh                          # prints the /etc/hosts line; add it, then open http://jupyter.datalab.test/
```

Jupyter password `123456`, S3 `seaweedadmin` / `seaweedadmin123` (local-dev credentials, same as compose).

### What `setup-minikube.sh` does (minikube NVIDIA tutorial, docker driver)

1. Checks `nvidia-smi`, `net.core.bpf_jit_harden=0` and the Docker `nvidia` runtime. It tells you the fix command instead of running `sudo`:
   ```bash
   echo "net.core.bpf_jit_harden=0" | sudo tee -a /etc/sysctl.conf && sudo sysctl -p
   sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
   ```
   If a minikube cluster existed before the NVIDIA runtime was installed, `minikube delete` it first.
2. `minikube start --driver docker --container-runtime docker --gpus all --kubernetes-version v1.37.0 ...`
   (`MINIKUBE_GPUS=nvidia.com` selects the CDI variant from the same tutorial). `--container-runtime docker` is required for the GPU and enables
   `eval $(minikube docker-env)` image builds. It also mounts `jupyter/notebooks` into the node at `/datalab/notebooks`.
3. Verifies `kubectl get nodes -o json | jq '.items[].status.capacity'` shows `nvidia.com/gpu` (enables the `nvidia-device-plugin` addon if not).
4. Creates namespaces `datalab` and `datalab-e2e` (Pod Security `warn`/`audit` at `baseline`).
5. Installs the Kubeflow Spark Operator 2.5.2 (watching both namespaces) and Envoy Gateway v1.9.2 (which also installs the Gateway API CRDs).

macOS and Windows cannot expose an NVIDIA GPU to minikube (see the tutorial); use `--gpu-mode none` there.

## Reaching the services

| URL (Gateway) | Compose equivalent |
|---|---|
| http://jupyter.datalab.test | localhost:8888 |
| http://trino.datalab.test | localhost:8080 |
| http://phoenix.datalab.test | localhost:6006 |
| http://ollama.datalab.test | localhost:11434 |
| http://qdrant.datalab.test | localhost:6333 |
| http://s3.datalab.test, http://filer.datalab.test, http://seaweed.datalab.test | localhost:8333, 8889, 9333 |
| http://spark.datalab.test | Spark UI of the Connect server (compose: localhost:8081 master UI) |

Not routed through the Gateway (as in compose, in-cluster only): Postgres x2, HMS Thrift/REST, Spark Connect gRPC, Qdrant gRPC, Phoenix OTLP.
`helm/scripts/port-forward.sh` forwards those to localhost (`--all` forwards the HTTP UIs on the compose ports as well).

## Spark and the notebooks

* `datalab-spark` (SparkConnect) runs **our** `datalab-playground/spark` image (Spark 4.1.3, hadoop-aws, Iceberg 1.12.0). The Iceberg REST catalog,
  S3A settings and `spark.sql.extensions` are set **server-side** (`spark.sparkConf` in `values.yaml`, mirrored from `spark/conf/spark-defaults.conf`):
  a Spark Connect client cannot set static confs such as `spark.sql.extensions`.
* Notebooks get `SPARK_REMOTE=sc://datalab-spark-server:15002`; `SparkSession.builder...getOrCreate()` then returns a Connect session.
  Do **not** call `.master(...)` together with `SPARK_REMOTE` (pyspark 4.1 raises `CANNOT_CONFIGURE_SPARK_CONNECT_MASTER`);
  `SparkContext`/RDD APIs do not exist under Spark Connect. The notebooks already follow both rules and still run on compose.
* The Kubeflow SDK is installed for `SparkClient().connect(base_url=...)`, `list_sessions()` and `get_session_logs()`. Its create mode,
  `submit_job()` and the `kubeflow[spark]` extra are **blocked** (hardcoded Spark 4.0.4 version/image and a `pyspark-connect==4.2.0` pin):
  see the blocker table in [docs/VERSIONS.md](../docs/VERSIONS.md).

## GPU modes (`gpu.mode`)

| Mode | What it renders | Cluster needs |
|---|---|---|
| `device-plugin` (default) | `resources.limits: nvidia.com/gpu: 1` on Ollama | `--gpus all` + `nvidia-device-plugin` addon (`setup-minikube.sh`) |
| `dra` (experimental) | `ResourceClaimTemplate` (`resource.k8s.io/v1`, DeviceClass `gpu.nvidia.com`) + `resourceClaims` | NVIDIA DRA driver (`setup-minikube.sh --gpu-mode dra`) |
| `none` | no GPU request (CPU) | nothing |

```bash
helm/scripts/deploy.sh --set gpu.mode=none      # also pass --set ollama.enabled=false to skip Ollama entirely
```

## Tests

```bash
helm/scripts/lint.sh                    # no cluster: parity check, helm lint, render 8 variants, missing-CRD guard
helm/scripts/lint.sh --server-dry-run   # + validate every render against the current cluster's API server and CRDs
helm/scripts/e2e-k8s.sh                 # the 19-check lakehouse e2e on the cluster (no GPU): --mode application|connect|both
```

`e2e-k8s.sh` deploys the lakehouse core into its own namespace (`datalab-e2e`), so a running platform is not disturbed.
`--mode application` has the Spark Operator `spark-submit` the script with our image; `--mode connect` runs it through Spark Connect
(the notebook path, needs the jupyter image).

## Configuration notes

* `values.yaml` pins every image to the compose tag; `check-parity.sh` fails if they, the Spark config or the e2e script drift apart.
* The local images are `imagePullPolicy: IfNotPresent` (Kubernetes would otherwise treat `:latest` as `Always` and try Docker Hub).
* `enableServiceLinks: false` on every pod: a Service named `phoenix` would otherwise inject `PHOENIX_PORT=tcp://...`, which Phoenix reads as its port setting.
* SeaweedFS has a **headless** Service (`seaweedfs`): its master/volume/filer/S3 processes must reach each other at that name on many ports.
  `seaweedfs-http` is the ClusterIP used by the Gateway.
* `values-minikube.yaml` lowers Trino's heap (`-Xmx4G` instead of the image's `-Xmx8G`) and nothing else about the services.
* Data stores are StatefulSets whose PVCs survive `helm uninstall` (`persistence.retainOnDelete`). Delete them with `kubectl -n datalab delete pvc --all`.
* If a Trino UI request through the Gateway is rejected because of `X-Forwarded-*` headers, add `http-server.process-forwarded=true` to `trino/etc/config.properties`.

## Layout

```
helm/
├── datalab/                 the chart (templates per service, values.yaml, values-minikube.yaml, files/e2e_lakehouse.py)
└── scripts/                 setup-minikube, build-images, deploy, lint, check-parity, e2e-k8s, hosts, port-forward, versions.env
```
