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
helm/scripts/setup-minikube.sh                 # cluster + Spark Operator + Envoy Gateway + Headlamp   (--gpu-mode none for CPU only)
helm/scripts/build-images.sh                   # builds the 4 local images inside minikube + pulls the pinned third-party images
helm/scripts/deploy.sh                         # installs the chart (values.yaml + values-minikube.yaml)
helm/scripts/gateway-forward.sh                # separate terminal: Envoy Gateway on localhost:8080 (no tunnel, sudo or /etc/hosts)
# open http://jupyter.datalab.localhost:8080/   and   http://headlamp.datalab.localhost:8080/  (token: helm/scripts/headlamp-token.sh)
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
6. Installs Headlamp 0.45.0 with the Kubeflow plugin (see below).

macOS and Windows cannot expose an NVIDIA GPU to minikube (see the tutorial); use `--gpu-mode none` there.

## Reaching the services

Everything HTTP goes through one Envoy Gateway listener; the hostname picks the service (`<name>.<gateway.domain>`, default domain `datalab.localhost`).
Two ways to reach the Gateway from the host:

| | Command | URLs | Needs |
|---|---|---|---|
| **A. port-forward (default)** | `helm/scripts/gateway-forward.sh [--port 8080]` | `http://<name>.datalab.localhost:8080/` | nothing else: `*.localhost` resolves to 127.0.0.1 in browsers and curl; no sudo, no tunnel, works the same on every OS |
| B. LoadBalancer | `minikube tunnel` (separate terminal) + `helm/scripts/hosts.sh` | `http://<name>.<domain>/` (port 80) | an `/etc/hosts` line (the script prints it; use `--set gateway.domain=datalab.test` for a non-`localhost` name) |

If `*.localhost` does not resolve on your machine (some non-systemd Linux setups, `requests`/`getaddrinfo` clients), run `helm/scripts/hosts.sh --forward` and add the printed `127.0.0.1 ...` line.

| Service (`<name>.datalab.localhost:8080`) | Compose equivalent |
|---|---|
| `jupyter` | localhost:8888 |
| `trino` | localhost:8080 |
| `phoenix` | localhost:6006 |
| `ollama` | localhost:11434 |
| `qdrant` | localhost:6333 |
| `s3`, `filer`, `seaweed` | localhost:8333, 8889, 9333 |
| `spark` | Spark UI of the Connect server (compose: localhost:8081 master UI) |
| `headlamp` | (new) Kubernetes UI with the Kubeflow plugin, see below |

Not routed through the Gateway (as in compose, in-cluster only): Postgres x2, HMS Thrift/REST, Spark Connect gRPC, Qdrant gRPC, Phoenix OTLP.
`helm/scripts/port-forward.sh` forwards those to localhost (`--all` forwards the HTTP UIs on the compose ports as well).

## Headlamp (Kubernetes UI) with the Kubeflow plugin

[Headlamp](https://headlamp.dev) 0.45.0 (official chart, its own Helm release in the `datalab` namespace) with the **Kubeflow plugin**
([`headlamp_kubeflow` 0.2.0-alpha](https://github.com/headlamp-k8s/plugins/tree/main/kubeflow)). `setup-minikube.sh` installs it (skip with `--no-headlamp`; install or upgrade alone with `helm/scripts/install-headlamp.sh`).

```bash
helm/scripts/headlamp-token.sh        # prints a login token (creates ServiceAccount headlamp-login with cluster-admin: local single-user clusters only)
# open http://headlamp.datalab.localhost:8080/ (after helm/scripts/gateway-forward.sh) and paste the token
```

* The chart's plugin-manager sidecar installs the plugin from Artifact Hub at pod start (`kubectl -n datalab logs deploy/headlamp -c headlamp-plugin`).
  It needs outbound access to `registry.npmjs.org`, `artifacthub.io` and `github.com`.
* The plugin detects which Kubeflow CRDs exist. Here that is the Spark Operator: **SparkApplication** and **ScheduledSparkApplication** get dedicated views
  (try `helm/scripts/e2e-k8s.sh` and watch `datalab-e2e`). The **SparkConnect** server (`v1alpha1`) is not covered by the plugin: use Headlamp's generic *Custom Resources* view.
  Notebooks/Katib/Pipelines/Trainer sections appear only if those Kubeflow components are installed (they are not part of this platform).
* The route is part of the datalab chart (`headlamp.enabled`, `gateway.routes`), the Deployment is not: Headlamp is a separate release (`helm/headlamp-values.yaml`, pins in `helm/scripts/versions.env`).

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

## Manual test checklist (first run on a real cluster)

The chart was validated against a real API server and its Spark pieces were run in our images, but **never on a live Kubernetes node**.
Work through this in order; each step says what "good" looks like and how to look closer when it is not.

| # | Step | Good looks like | If not, look at |
|---|---|---|---|
| 1 | `helm/scripts/lint.sh` | `all static checks passed` | the message names the failing variant |
| 2 | `helm/scripts/setup-minikube.sh --gpu-mode none` (CPU first; add the GPU later) | ends with `Cluster ready`; `kubectl get pods -A` all Running | `kubectl -n spark-operator get pods`, `kubectl -n envoy-gateway-system get pods` |
| 3 | `helm/scripts/build-images.sh` | `minikube image ls` lists the 4 `datalab-playground/*` images | the Jupyter build needs `quay.io`; a corporate proxy/CA usually shows here |
| 4 | `helm/scripts/e2e-k8s.sh --mode connect` | `connect mode: 19/19` (this path was run in Docker already) | `kubectl -n datalab-e2e get pods,sparkconnect`; `kubectl -n datalab-e2e logs sparkconnect-pod...` |
| 5 | `helm/scripts/e2e-k8s.sh --mode application` | `application mode: 19/19` | `kubectl -n datalab-e2e describe sparkapplication datalab-e2e`; the operator's 4.0.4 `spark-submit` against our 4.1.3 image is the least-tested link |
| 6 | `helm/scripts/deploy.sh --set gpu.mode=none --set ollama.enabled=false` | `helm status datalab` deployed; all pods Ready | `kubectl -n datalab get pods`; `kubectl -n datalab describe pod <name>` |
| 7 | `helm/scripts/gateway-forward.sh`, then open Jupyter | http://jupyter.datalab.localhost:8080 loads; a cell runs (WebSocket works); the last notebook cell lists `datalab-spark` | `kubectl -n datalab get gateway,httproute`; if Trino rejects proxied requests see *Configuration notes* |
| 8 | Run `data_lab_playground.ipynb` | no `CANNOT_CONFIGURE_SPARK_CONNECT_MASTER`; Iceberg/Trino cells pass | Jupyter pod logs; `kubectl -n datalab logs datalab-spark-server` |
| 9 | Save a notebook | the file appears in `jupyter/notebooks/` on the host | hostPath permission: switch to `--set jupyter.notebooks.mode=pvc` |
| 9b | Headlamp: `helm/scripts/headlamp-token.sh`, open http://headlamp.datalab.localhost:8080 | login works; the sidebar has a **Kubeflow** section with *Spark* entries; `datalab-spark` visible under Custom Resources | `kubectl -n datalab logs deploy/headlamp -c headlamp-plugin` (plugin install: check `curl -s https://artifacthub.io/api/v1/packages/headlamp/headlamp-plugins/headlamp_kubeflow` returns version 0.2.0-alpha); browser console for plugin load errors |
| 10 | GPU: `setup-minikube.sh` (device plugin) then `deploy.sh`; wait for `job/ollama-pull-*` | `kubectl -n datalab exec deploy/ollama -- nvidia-smi` shows the GPU; `rag_demo.ipynb` runs | `kubectl describe node | grep nvidia`; `kubectl -n datalab describe pod -l app.kubernetes.io/name=ollama` |
| 11 | (optional) `--gpu-mode dra` | Ollama pod has a bound `ResourceClaim` | `kubectl -n datalab get resourceclaim`; DRA is experimental here |

Useful everywhere: `kubectl -n datalab get pods -w`, `kubectl -n datalab get events --sort-by=.lastTimestamp | tail -20`,
`helm -n datalab get values datalab --all`. Reset everything: `helm -n datalab uninstall datalab && kubectl -n datalab delete pvc --all`.

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
├── headlamp-values.yaml     values for the official Headlamp chart (+ Kubeflow plugin)
├── datalab/                 the chart (templates per service, values.yaml, values-minikube.yaml, files/e2e_lakehouse.py)
└── scripts/                 setup-minikube, build-images, deploy, install-headlamp, headlamp-token, gateway-forward, hosts, port-forward, lint, check-parity, e2e-k8s, versions.env
```
