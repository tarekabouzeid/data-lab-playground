# Helm chart gap report

Audit date: 2026-10-12, against `main` @ `a4d8a53`. The fixes are in `charts/data-lab` and
`.github/workflows/chart-*.yaml`; see "Verification" at the end for what was proven where.

## Finding 0: there is no Helm chart

The repo has no `charts/` directory, `Chart.yaml`, `values.yaml` or any Kubernetes manifest. The platform
is Docker Compose only. So every requirement below starts as **missing**, and "fix" means building it.

Three of the requested components do not exist in the Compose stack at all:
- **Spark Connect**: Compose runs a standalone master/worker.
- **Airflow**: not in Compose.
- **Executor RBAC**: there are no Kubernetes executors in Compose.

These are new in the chart only. Per "don't touch the Docker Compose stack except to keep versions
aligned", they are **not** added to Compose.

## Requirement → status → fix

| # | Requirement | Status | Fix |
|---|---|---|---|
| 1a | Chart at `charts/data-lab`, semver `version`, `appVersion` = lab release | ❌ missing | New chart `version: 0.1.0`, `appVersion: "3.0.0"` (see C9) |
| 1b | CI: helm lint, ct lint, kubeconform; OCI publish to ghcr.io on tag | ❌ missing | `.github/workflows/chart-lint.yaml`, `chart-release.yaml` (images + chart on `v*` tag) |
| 1c | `values.schema.json` for all values; helm-docs table in README | ❌ missing | Hand-written strict schema (`additionalProperties: false`); `README.md.gotmpl` + helm-docs, CI fails on drift |
| 2a | `<component>.enabled` for hms, seaweedfs, spark, trino, airflow, jupyter, ollama, qdrant, phoenix | ❌ missing | Toggles in values; ollama/qdrant/phoenix/jupyter `false` by default |
| 2b | Invalid combos fail at render | ❌ missing | `templates/validate.yaml` with `fail` (trino/spark without hms, hms without S3, …) |
| 3a | Names from release fullname; no hardcoded names/namespaces | ❌ missing (Compose hardcodes `hive-metastore`, `seaweedfs`, …) | Every name is `<fullname>-<component>`, every host is a release service; namespace is `.Release.Namespace` |
| 3b | No cluster-scoped resources; executor RBAC = Role/RoleBinding | ❌ missing | Only namespaced kinds; CI asserts it on the rendered output |
| 3c | Two releases in one namespace and in two namespaces | ❌ missing | Name isolation + selectors use `app.kubernetes.io/instance`; CI installs both cases |
| 4 | No default passwords; `existingSecret` everywhere; else generated and stable | ❌ missing (Compose hardcodes all credentials) | Per-credential `existingSecret` + generated Secret using `lookup` to reuse values on upgrade (see C6) |
| 5a | `global.imageRegistry`, `imagePullSecrets`, tags pinned to VERSIONS.md | ❌ missing | Image helper; pins listed in `docs/VERSIONS.md` §8 |
| 5b | Per-component resources, storageClass, persistence on/off (emptyDir) | ❌ missing | Common per-component block |
| 5c | runtimeClassName, nodeSelector, tolerations, podLabels/annotations | ❌ missing | Common per-component block, `global.*` fallbacks |
| 6a | runAsNonRoot, drop ALL, seccomp RuntimeDefault, no privileged; document PSS exceptions | ❌ missing | Restricted security context on every pod, including Spark executors via a pod template |
| 6b | ClusterIP only; `app.kubernetes.io/*` labels everywhere | ❌ missing | `type: ClusterIP` hardcoded; common label helper (executors get them through Spark conf) |
| 7a | ConfigMap `<fullname>-endpoints` with endpoints + credential Secret names | ❌ missing | `templates/endpoints-configmap.yaml` |
| 7b | Airflow: LocalExecutor; DAGs from a PVC or git-sync | ❌ missing (no Airflow at all) | Airflow 3.3.2 single pod (api-server, scheduler, dag-processor, triggerer), own Postgres, `dags.source: pvc\|gitSync\|none` |
| 8a | Readiness/startup probes; HMS waits for DB and S3 | ❌ missing (Compose only has `depends_on`) | Probes on every container; HMS init containers wait for Postgres and S3 and create the bucket |
| 8b | `helm test`: Spark creates an Iceberg table, Trino reads it, cleanup | ❌ missing | Test pod: PySpark Connect client → Iceberg table → Trino REST query → drop |
| 8c | Minimal profile ready < 3 min on kind; footprint per component documented | ❌ missing | `ci/minimal-values.yaml`; CI times `helm install --wait`; footprint table in chart README |
| 9 | CI e2e on kind (default, minimal, 2 parallel, helm test, uninstall leaves nothing) | ❌ missing | `.github/workflows/chart-e2e.yaml` |

## Conflicts and component limitations (reported, not worked around)

| ID | Conflict | Decision |
|---|---|---|
| C1 | **HMS, Spark and Jupyter images are custom and not published anywhere.** The chart cannot pull `datalab-playground/*:latest`. | CI builds them. The release workflow pushes them to `ghcr.io/tarekabouzeid/data-lab-playground/*:<appVersion>`. **Until the first `v3.0.0` tag is pushed, chart defaults point to images that do not exist yet.** The kind e2e builds them and loads them into kind. |
| C2 | **Some endpoints have no auth mechanism in this setup.** <br>• HMS Thrift has no auth without Kerberos. <br>• HMS REST supports only `none`/`jwt`. <br>• Trino password auth requires HTTPS plus a shared secret. <br>• Ollama has no auth at all. | "No default passwords" holds because these have no password. Isolation comes from a **NetworkPolicy** (on by default) that admits only pods of the same release, plus namespaces you list. Credentials that do exist get Secrets: S3, Postgres, Spark Connect token, Airflow, Jupyter, Qdrant, Phoenix. |
| C3 | **Ollama needs an NVIDIA GPU to be useful.** | `ollama.gpu.enabled` adds the `nvidia.com/gpu` limit and an optional `runtimeClassName`. It runs on CPU otherwise (slow). It is off by default and not exercised in CI (no GPU on kind). |
| C4 | **Postgres 13 is EOL.** Compose `metastore-db` still uses it, and moving Compose to 17 breaks existing data dirs. | The chart uses `postgres:17.11` for HMS and Airflow. Compose stays on 13. The divergence is documented in VERSIONS.md. |
| C5 | **Spark Connect Python client deps are missing from the Spark image** (pandas, pyarrow, grpcio, …). The `helm test` and Jupyter-over-Connect both need them. | Added pinned wheels to `spark/Dockerfile`. This is additive and Compose behaviour is unchanged. |
| C6 | **Generated secrets use `lookup`.** It returns nothing under `helm template`, `--dry-run=client`, Argo CD and Flux's post-render. | Under plain `helm install/upgrade` (and the Helm Go SDK Sandpiper uses), values are stable across upgrades. Template-only consumers **must** set `existingSecret`. This is documented. |
| C7 | **No Kubernetes cluster can run in the authoring sandbox** (nested runc cannot set `oom_score_adj`). | Local checks cover helm lint/template, kubeconform, schema, and the containers' non-root commands in Docker. **The kind e2e in GitHub Actions is the authoritative check.** |
| C8 | **Airflow is new.** There is no Compose service to align with, and adding one would touch Compose. | The chart only pins `apache/airflow:3.3.2-python3.12`. Its metadata DB is a dedicated Postgres. The default auth manager is SimpleAuthManager with one admin user from a Secret. |
| C9 | **`appVersion` must equal the lab release, but the current lab state (Spark 4.1.3 / Iceberg 1.12.0 / HMS 4.2.1 / Trino 483) is not released yet.** `RELEASE_NOTES.md` ends at v2.0.0. | Used `3.0.0`, the release already suggested for this matrix. Writing `RELEASE_NOTES.md` still needs your confirmation (policy). The chart has its own `charts/data-lab/CHANGELOG.md`. |
| C10 | **PSS "restricted" exceptions.** | None in the default or minimal profile. Every image runs with a numeric non-root UID. SeaweedFS's entrypoint only chowns when root, so that step is skipped. Postgres runs as 999. Ollama runs as 1000 with `HOME` on a volume. Any exception found by CI is recorded in the chart README. |
| C11 | **The Spark Connect token cannot use the standard `;token=` URL.** The PySpark client sends a token only over TLS (plaintext only to `localhost`), and the Spark 4.1 Connect server has no TLS. | Auth stays on by default. Clients pass the token as gRPC metadata, `sc://host:15002/;authorization=Bearer <token>`, which the stock client supports. Verified: right token works, wrong or missing token is rejected. It is published as `connectUrlTemplate` in the endpoints ConfigMap. The alternative is a TLS proxy sidecar (not built). |
| C12 | **Ollama was not run-tested.** The image (CUDA libs, ~9 GB unpacked) did not fit in the authoring sandbox, and there is no GPU on kind. | Render, schema and policy checks only. The non-root setup (`HOME`/models on the volume) follows the image's documented behaviour but is unverified. |
| C13 | **The Semantic Operator server trusts identity headers.** Its `header` auth mode takes `X-Semantic-User`/`X-Semantic-Role` as sent; the alternative is `jwt`, which needs an identity provider (JWKS). | `semantic.server.auth.mode: header` is the default, safe only because the release NetworkPolicy admits nothing but this release's pods (same posture as C2). `jwt` + `semantic.server.auth.jwt.jwksURL` is exposed for shared clusters; rendering fails for `jwt` without a JWKS URL. |
| C14 | **The Semantic Operator manager has no label selector.** It reconciles every SemanticModel in its namespace, and the server serves every compiled model there. | At most one semantic-enabled release per namespace. Its manager Role is upstream's: ConfigMap create/update/delete across the release namespace (where it publishes compiled models), so do not share that namespace with untrusted workloads. A `lookup` guard in `templates/validate.yaml` refuses a second one at install time (skipped under `helm template`, where `lookup` is empty). Releases without `semantic` are unaffected, so the "two releases per namespace" requirement holds by default. |
| C15 | **The SemanticModel CRD is cluster-scoped** and the upstream project has no GitHub release yet (chart 0.1.0 in the repo, images `v0.1.1` on ECR Public). | The chart never ships the CRD: it is vendored in `hack/crds/` (commit `4fd7057`) and applied by the platform admin, the kind e2e or `hack/workstation/setup-minikube.sh`. Rendering fails with a hint when the API is missing (`global.apiChecks`). |
| C16 | **The manager puts a finalizer on every SemanticModel**, so `helm uninstall` (which removes the manager together with the models) can leave models stuck and the namespace undeletable. | A `pre-delete` hook Job (stdlib Python in the Spark image, namespaced Role on `semanticmodels`) deletes the release's models while the manager still runs and strips the finalizer after 90 s. |
| C17 | **Ossie has no `TRINO` dialect yet** (apache/ossie#607 is open) and Iceberg views through the HMS REST catalog are unverified. | The dbt-demo model uses `ANSI_SQL` expressions only and no `spec.views`. |
| C18 | **The workstation lab needs cluster-scoped pieces** (Spark Operator, Envoy Gateway's GatewayClass, a host-path PersistentVolume, the GPU plugin or DRA driver, Headlamp with cluster-admin). | They stay out of the chart: `hack/workstation/setup-minikube.sh` and `install-headlamp.sh` install them. The chart only references them (`gateway.gatewayClassName`, `jupyter.persistence.existingClaim`, `spark.applications`), so every profile renders namespaced and Pod Security "restricted" (`tests/chart/test_workstation_profile.py`). |

## Verification

| Check | Where | Result |
|---|---|---|
| helm lint `--strict`, ct lint (yamllint + yamale), kubeconform `-strict` on 5 profiles | local + `chart-lint` CI | ✅ |
| No cluster-scoped kinds, single namespace, `<fullname>-` names, app.kubernetes.io labels, ClusterIP only, PSS restricted on every pod/init/test/executor template | `hack/check-chart-render.py`, local + CI | ✅ |
| Invalid combos and unknown keys fail at render | local + CI | ✅ |
| SeaweedFS as 1000 (all caps dropped): server + postStart bucket creation | Docker, rendered script | ✅ ~10 s |
| HMS as 1000 with rendered templates and secrets from env: schema init on Postgres, Iceberg REST up; wait-for-db and wait-for-s3 (wrong key keeps waiting) | Docker | ✅ ready in 27 s |
| Trino as 1000 with rendered config and `${ENV:...}` S3 keys | Docker | ✅ ready in 59 s |
| `helm test` script: Spark Connect (token) → Iceberg table via HMS REST → SeaweedFS → Trino reads `[[3, 6]]` → table purged and namespace dropped | Docker (Connect server in `local` mode) | ✅ 32 s |
| Airflow as 50000:0: auth-file init, `db migrate`, api-server + scheduler + dag-processor, admin login (wrong password → 401), DAG from read-only folder, LocalExecutor run → success | Docker | ✅ |
| Qdrant (API key enforced, health open), Phoenix (auth on, admin login from Secret), Postgres 17.11 as 999 | Docker | ✅ |
| kind e2e `default` (Airflow on): install, endpoints contract (every listed Secret exists), 1 Spark executor pod, `helm test` passed, generated Secrets unchanged after `helm upgrade`, uninstall leaves the namespace empty | `chart-e2e` run 38301197335 | ✅ |
| kind e2e `minimal`: Ready in **29 s** (budget 180 s, images pre-loaded), executor pod, `helm test` passed, namespace empty after uninstall | `chart-e2e` run 38301197335 | ✅ |
| kind e2e `parallel`: `p1` + `p2` in one namespace and `lab` in two namespaces installed at the same time, all four `helm test` passed, a `p2` pod cannot reach `p1`'s Trino (NetworkPolicy enforced), all three namespaces empty after uninstall | `chart-e2e` run 38301197335; locally on WSL2 with `hack/kind-up.sh --calico` (2026-10-27) | ✅ |
| git-sync, Jupyter, Ollama | render only | ⚠️ not run (registry.k8s.io / quay.io blocked in the sandbox; Ollama see C12) |
| kind e2e `default` with both MCPs on: Airflow task log readable after an Airflow pod restart; Spark History Server lists the Connect app as completed (7 jobs) after a graceful restart; Spark History MCP answers `history` and `live`; Airflow MCP lists DAGs and its `trigger_dag` is refused (read-only, `viewer` user); namespace empty after uninstall | `chart-e2e` run 38355323212 | ✅ |
| Merge of `helm/datalab` (2026-10-30): default values render the same resources as 0.1.0 (golden list); every `ci/` profile namespaced, PSS restricted, kubeconform-valid (Gateway kinds from the CRDs catalog); Gateway, DRA, SparkApplication RBAC, lakehouse helm test, Semantic Operator, SemanticModel (validated against the vendored CRD), uninstall hook, workstation profile | `tests/chart` (pytest, 80 tests), `chart-lint` CI | ✅ |
| kind e2e `semantic`: model unpublished before its tables exist, published after, `total_revenue == 40` over MCP (returned order excluded), `marketing` refused `stg_customers.email`, second semantic release refused (C14), clean uninstall | `chart-e2e` (see the run on the branch head) | ⏳ first run pending (the previous run hit ECR Public's anonymous pull limit, fixed in the workflow) |
| Workstation lab on minikube with the merged chart | `docs/WORKSTATION_TESTING.md` (manual) | ⏳ to be run on the workstation |
