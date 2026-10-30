# Unified Helm chart + Semantic Operator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One Helm chart (`charts/data-lab`) that stays portable for Sandpiper and also covers the workstation lab (minikube + GPU) that `helm/datalab` does today. Add the Kubeflow Semantic Operator on top, plus a "chat with your data" notebook.

**Architecture:** `charts/data-lab` is the base. Its defaults and guarantees do not change: namespaced only, Pod Security "restricted", generated or `existingSecret` credentials, several releases per namespace. The useful parts of `helm/datalab` come in as opt-in values: Gateway routes, Ollama GPU modes, notebooks from a host folder, Spark Operator integration, the 19-check lakehouse e2e. A `ci/workstation-values.yaml` profile turns them on. Anything cluster-scoped is installed by workstation scripts under `hack/workstation/`, never by the chart. This covers the Spark Operator, Envoy Gateway and its GatewayClass, the NVIDIA plugin or DRA driver, Headlamp, the Semantic Operator CRD and the host-path PersistentVolume. The Semantic Operator's manager and server are rendered by our chart, namespaced, and point at the release's Trino. `helm/` is deleted at the end.

**Tech Stack:** Helm 3.19+ (chart `apiVersion: v2`), Kubernetes ≥ 1.29 (DRA needs ≥ 1.34), kind (CI), minikube 1.39 (workstation), Kubeflow Spark Operator 2.5.2, Envoy Gateway v1.9.2, Headlamp 0.45.0, Kubeflow Semantic Operator (images `v0.1.1`, CRD from commit `4fd7057`), Apache Ossie model format, Ollama 0.40.3, Python 3.12 (`pytest`, `mcp`, `ollama` clients).

**Spec:** this document, section "Design". There is no separate spec. The decisions below come from the user's requests in the session on 2026-10-30 ("merge these charts", "add the semantic operator", "ensure that the chart is portable for Sandpiper", "freeze compose"), from the minikube real-node run recorded in `docs/K8S_HELM_PLAN.md` (Addendum 2026-10-27), and from the semantic-operator sources at commit `4fd7057`.

---

## Design

### D1. What survives from each chart

| Concern | Merged chart | Source | Why |
|---|---|---|---|
| Spark Connect | Chart-owned Deployment (`exec spark-submit`), token auth | `charts/data-lab` | Portable, no operator needed, closes event logs for the History Server. The `SparkConnect` CR is dropped (the operator ran `start-connect-server.sh`, and `helm --wait` ignores CRs). |
| Spark batch jobs | Optional `spark.applications`: ServiceAccount + Role for `SparkApplication` drivers. The operator itself is a prerequisite. | `helm/datalab` | The real-node run verified `SparkApplication` (19/19), and the Headlamp Kubeflow plugin only shows SparkApplications. |
| Credentials | Generated or `existingSecret` | `charts/data-lab` | Sandpiper requirement. Notebooks read them from env (D4). |
| Browser access | Optional Gateway API `HTTPRoute`s. Either a namespaced `Gateway` on an existing GatewayClass, or routes attached to an existing shared Gateway (`parentRef`). | `helm/datalab` | Same UX as today's `*.datalab.localhost:8080`, and Sandpiper can attach to its own Gateway. The GatewayClass (cluster-scoped) moves to the setup script. |
| Notebooks from the host | `jupyter.persistence.existingClaim` bound to a host-path PersistentVolume that the workstation script creates | `helm/datalab` idea | A `hostPath` volume in the pod breaks Pod Security "restricted". A PV is cluster-scoped, so the script creates it and the chart only claims it. |
| GPU for Ollama | `ollama.gpu.enabled` (device plugin, existing) plus new `ollama.gpu.dra` (namespaced `ResourceClaimTemplate`) | both | DRA stays optional and is still not verified on a real node. |
| Headlamp | A separate release installed by a script (it needs cluster-admin). The chart only routes to it via `gateway.extraRoutes`. | `helm/datalab` | Cluster-scoped, so it must stay out of the chart. |
| Lakehouse e2e (19 checks) | Optional helm test `tests.lakehouseE2e`, run over Spark Connect | `helm/datalab` | A stronger test than the round-trip one. It runs on kind and on the workstation. |
| Trino behind a proxy | `http-server.process-forwarded=true` whenever `gateway.enabled` | `38f7655` | Without it the Gateway gets HTTP 406. |

### D2. Sandpiper portability guarantees (unchanged, now enforced for every profile)
- With default values, the rendered resource set is identical to chart 0.1.0, apart from label and annotation changes from the version bump.
- No cluster-scoped kind is rendered under any profile in `ci/`. `hack/check-chart-render.py` runs on all of them.
- Every pod passes Pod Security "restricted" under every profile.
- New values keys are additions only. Each has a `# --` doc comment. The schema stays generated and closed.
- Two releases in one namespace, and in two namespaces, keep working. Gateway hostnames include the release name. **Exception (C14):** at most one release with `semantic.enabled` per namespace (D3).

### D3. Semantic Operator inside the chart
- **CRD** `semanticmodels.semantic.ossie.io` (`v1alpha1`, namespaced kind, but the CRD itself is cluster-scoped). It is vendored at `hack/crds/semantic.ossie.io_semanticmodels.yaml` from commit `4fd7057`. It is applied by `hack/workstation/setup-minikube.sh`, by the kind e2e, and by Sandpiper's platform admin, never by the chart.
- **The chart renders, when `semantic.enabled`:**
  - ServiceAccounts and namespaced Roles. The manager may get/list/watch/update SemanticModels and their status, write ConfigMaps, and manage the leader-election Lease. The server may get/list/watch ConfigMaps.
  - Config ConfigMaps for the manager and the server.
  - A manager Deployment, a server Deployment (1 replica) and a ClusterIP Service on 8090 (`/mcp`, `/v1/*`).
  - A NetworkPolicy, covered by the release-wide policy.
  - The `SemanticModel` objects from `semantic.models`.

  The config file layout and container args are copied from the upstream chart at `4fd7057` (`charts/semantic-operator/templates/{configmap,manager-deployment,server-deployment,rbac}.yaml`), not invented. Our chart's names, labels, security context and probes are applied on top.
- **Engine:** `type: trino`. The host is the release's Trino Service on port 8080. The users are `semantic-manager` and `semantic-server`. There is no password and no TLS: our Trino has no auth, which is C2 in `docs/HELM_GAP_REPORT.md`.
- **Auth:** the default is `header` mode with `allowInsecureHeaderAuth: true`. This is only safe because the release NetworkPolicy admits only same-release pods (recorded as C13). `semantic.server.auth.mode: jwt` and its `jwt.*` keys are exposed for Sandpiper.
- **One semantic-enabled release per namespace (C14):** upstream has no label selector. A manager reconciles every SemanticModel in its namespace, and a server serves every compiled ConfigMap in it. `templates/validate.yaml` uses `lookup` to fail when another release's semantic manager Deployment exists in the namespace. Under `helm template`, `lookup` is empty, so the check is skipped and the limit is documented.
- **API check:** when `semantic.enabled` and `.Capabilities.APIVersions.Has "semantic.ossie.io/v1alpha1/SemanticModel"` is false, rendering fails with a hint to apply the CRD. CI passes `--api-versions semantic.ossie.io/v1alpha1/SemanticModel`.
- **Model `dbt-demo`** (on by default when `semantic.enabled`) covers Trino catalog `iceberg`, schema `dbt_demo`. These are the dbt demo's tables (`fct_orders`, `stg_customers`, `dim_products`).
  - All expressions use dialect `ANSI_SQL`, because `TRINO` is not in the CRD enum yet (apache/ossie#607 is open).
  - No `spec.views`: Iceberg view support in the HMS REST catalog is unverified, so views are out of scope.
  - Metrics:
    - `total_revenue` = `SUM(CASE WHEN status <> 'returned' THEN amount ELSE 0 END)`. Returns are excluded on purpose: this is the trap raw text-to-SQL falls into.
    - `order_count` = `COUNT(DISTINCT order_id)`
    - `units_sold` = `SUM(quantity)`
    - `average_order_value` = `total_revenue / order_count`, written as one ANSI expression.
    - `customer_count` = `COUNT(DISTINCT customer_id)`
  - Dimensions: `country`, `plan` (customers); `category`, `product_name` (products); `order_date` (`order_ts`, `is_time: true`); `status`.
  - Relationships: `fct_orders.customer_id → stg_customers.customer_id` and `fct_orders.product_id → dim_products.product_id`.
  - Governance roles: `analyst` (all metrics and fields), `marketing` (all metrics, `denyFields: [email]`). The default role is `analyst`.
- **Resync:** `semantic.manager.resyncPeriod: 30s`, so a model applied before `dbt build` becomes Ready within about 30 s of the tables appearing.
- **Endpoints ConfigMap keys:** `SEMANTIC_MCP_URL` = `http://<fullname>-semantic:8090/mcp`, `SEMANTIC_REST_URL` = `http://<fullname>-semantic:8090/v1`, `SEMANTIC_DEFAULT_ROLE` = `analyst`.

### D4. Notebooks on the merged chart
- Jupyter gets every variable that `jupyter/notebooks/lab_utils.py` and `rag_demo.ipynb` read, built from the release's Services and Secrets. That includes `TRINO_HOST`, `TRINO_PORT`, `S3_ENDPOINT`, `S3_ACCESS_KEY`, `S3_SECRET_KEY` and `ICEBERG_REST_URI`. It also gets the Ollama, Qdrant and Phoenix variables their code reads, `SPARK_REMOTE` = `sc://<fullname>-spark:15002/;authorization=Bearer $(SPARK_CONNECT_TOKEN)`, and the `SEMANTIC_*` keys.
- New notebook `jupyter/notebooks/semantic/01_chat_with_your_data.ipynb`. It has no `.master()`, no `DROP SCHEMA`, and no module named `dbt_*`, per AGENTS.md pitfall 18. It runs:
  1. Stop with a clear message when `SEMANTIC_MCP_URL` is unset (Compose, or a chart without semantic).
  2. Run `lab_utils.dbt("build")` when `iceberg.dbt_demo.fct_orders` is missing.
  3. Wait for MCP `list_models` to contain `dbt-demo` (timeout 180 s).
  4. Ask 5 fixed questions in two ways: the **semantic path** (Ollama tool-calling over the MCP tools `list_models`, `list_metrics`, `list_dimensions`, `query_metric`) and the **raw path** (the same model gets the table DDL and writes Trino SQL).
  5. Compute ground truth with hand-written SQL, show tool calls, the compiled SQL, both answers and a score table.
- Tool-calling model: `qwen3:4b` (gemma3 has no tool calling in Ollama). It is added to `ollama.models` in the workstation profile. The notebook checks `ollama show` capabilities for `tools` and stops with a message if the capability is missing.
- Helpers live in `jupyter/notebooks/lab_utils.py` with unit tests in `tests/unit/test_lab_utils_semantic.py`.

### D5. Docker Compose is frozen
Compose keeps its current services and only gets version alignment. It does not get the Semantic Operator, Airflow, Spark History or the MCP servers. Notebooks that need those detect their absence (D4 step 1). README and AGENTS.md say so.

### D6. Workstation tooling
`helm/scripts/*` becomes `hack/workstation/*`, keeping behaviour and verified fixes. It points at `charts/data-lab` with `ci/workstation-values.yaml` and release `datalab` in namespace `datalab`, so the URLs stay `jupyter.datalab.localhost:8080` (hostTemplate `{{route}}.{{release}}.{{domain}}`, domain `localhost`). The scripts are `setup-minikube.sh`, `build-images.sh`, `deploy.sh`, `e2e.sh`, `install-headlamp.sh`, `headlamp-token.sh`, `gateway-forward.sh`, `port-forward.sh` and `versions.env`. `hack/kind-up.sh` stays. `helm/` and `helm/scripts/check-parity.sh` are deleted. Parity with Compose is kept by `hack/check-compose-parity.py` (D7).

### D7. Compose parity check (replaces `check-parity.sh`)
`hack/check-compose-parity.py` fails when either of these holds:
- A non-credential, non-master key in `spark/conf/spark-defaults.conf` is missing from, or different in, the chart's rendered `spark-defaults.conf`. The allowed differences are listed in the script with a reason.
- `charts/data-lab/files/e2e_lakehouse.py` differs from `tests/e2e/e2e_lakehouse.py`.

It runs in `chart-lint` CI.

## Global Constraints

- Develop only on branch `claude/upgrade-iceberg-spark-hive-trino-yc0p1s`; never create a PR unless asked.
- Chart version `0.1.0` → `0.2.0`; `appVersion` stays `"3.0.0"`; custom image tags = `appVersion`.
- `values.yaml` is a public API: add keys only, never rename/remove; every key has a `# --` comment; regenerate `values.schema.json` with `hack/gen-chart-schema.py` and README with `helm-docs -c charts -s file`.
- Names only from `datalab.componentName`; labels from the existing helpers; Service type `ClusterIP` only.
- No cluster-scoped kind rendered under any values file in `charts/data-lab/ci/`.
- Pod Security "restricted" for every pod/init/test container; no `hostPath`, `hostNetwork`, privileged, or root UID.
- Semantic Operator images: `public.ecr.aws/data-on-eks/semantic-operator/manager:v0.1.1` and `public.ecr.aws/data-on-eks/semantic-operator/server:v0.1.1`; CRD and config layout from `kubeflow/semantic-operator` commit `4fd7057`.
- Ossie dialect `ANSI_SQL` only.
- Tool-calling model `qwen3:4b`; existing models unchanged.
- Compose stack unchanged apart from version alignment.
- Never write `RELEASE_NOTES.md` without asking the user: *"Is this feature/change complete and ready to be documented in RELEASE_NOTES.md for the next release?"*
- Commits end with the session's `Co-Authored-By` / `Claude-Session` trailers; no model identifiers in commits.
- Post-change rule: README, AGENTS.md, CLAUDE.md, docs/VERSIONS.md stay current (Task 10).

## Review Focus

1. **Two semantic-enabled releases in one namespace.** Expected: the second `helm install` fails with a clear message, and the two managers never both reconcile. The test is in Task 5 (`test_semantic_second_release_rejected`, kind e2e step).
2. **A SemanticModel applied before the dbt tables exist.** Expected: the model shows not-Ready, then Ready within about 30 s of `dbt build`. The notebook waits and does not crash. Tests are in Task 6 (kind e2e: seed after apply) and Task 8 (wait helper with timeout).
3. **Default values render exactly what 0.1.0 rendered.** Expected: a Sandpiper upgrade from 0.1.0 creates and deletes nothing. The golden-list test is in Task 1.
4. **Gateway on with the NetworkPolicy on.** Expected: the Envoy proxy in another namespace reaches the Services, and other namespaces still can't. The render test is in Task 2.
5. **Notebook on a chart where Spark Connect auth is on.** Expected: `SPARK_REMOTE` carries the bearer token, and a session works without user setup. The render test is in Task 1, and the workstation e2e runs `iceberg/00`.

---

### Task 1: Render-test harness, golden default render, Jupyter env for notebooks

**Files:**
- Create: `tests/chart/conftest.py`, `tests/chart/test_defaults.py`, `tests/chart/test_jupyter.py`, `tests/chart/golden/default-resources.txt`
- Modify: `charts/data-lab/templates/jupyter.yaml`, `charts/data-lab/values.yaml` (`jupyter.persistence.existingClaim`), `charts/data-lab/Chart.yaml` (version `0.2.0`)

**Interfaces:**
- Produces: `render(values: dict | None = None, profile: str | None = None, release="r", namespace="ns", api_versions: list[str] = ()) -> list[dict]` in `tests/chart/conftest.py` (runs `helm template`, returns parsed docs; skips the test if `helm` is absent); `by_kind(docs, kind) -> list[dict]`; `env_of(docs, component) -> dict[str, dict]` (container env by name, from the Deployment/StatefulSet whose `app.kubernetes.io/component == component`).

- [ ] **Step 1: Write the golden default test.** `test_default_resource_set_unchanged`: `sorted(f"{d['kind']}/{d['metadata']['name']}" for d in render())` equals the lines of `golden/default-resources.txt`. Generate the golden file from the **0.1.0** chart (`git show c18e8b7:charts/data-lab` → temp dir → `helm template r … -n ns`) and commit it.
- [ ] **Step 2: Write the Jupyter env tests.** `test_jupyter_env_matches_lab_utils`: with `{"jupyter": {"enabled": True}}`, `env_of(docs, "jupyter")` plus the endpoints ConfigMap keys contain every name read by `os.environ.get(...)` in `jupyter/notebooks/lab_utils.py` (parse the file with a regex in the test) and `SPARK_REMOTE`. `test_spark_remote_has_bearer`: `SPARK_REMOTE` value contains `;authorization=Bearer $(SPARK_CONNECT_TOKEN)` and `SPARK_CONNECT_TOKEN` comes from the Spark Connect Secret. `test_existing_claim`: with `jupyter.persistence.existingClaim: nb`, the pod's data volume uses `persistentVolumeClaim.claimName: nb` and no PVC is rendered for Jupyter.
- [ ] **Step 3: Run the tests and confirm they fail.** Run `pytest tests/chart -q`. Expected: the golden test passes, and the jupyter tests FAIL (missing env, missing key).
- [ ] **Step 4: Implement.** Add the env entries in `templates/jupyter.yaml`, using `datalab.host` for hosts and `secretKeyRef` for credentials. Add the `jupyter.persistence.existingClaim` key (`# -- Use an existing PVC for /home/jovyan/work (e.g. the workstation host-path PV); no PVC is created.`) and honour it in the `datalab.simpleComponent` call. Bump Chart.yaml `version: 0.2.0`.
- [ ] **Step 5: Run the tests and lint.** Run `pytest tests/chart -q && helm lint --strict charts/data-lab --kube-version 1.34.0 && python3 hack/gen-chart-schema.py && helm-docs -c charts -s file`. Expected: all pass.
- [ ] **Step 6: Commit.** Message: `chart: render tests, golden default set, notebook env for Jupyter`.

### Task 2: Gateway routes

**Files:**
- Create: `charts/data-lab/templates/gateway.yaml`, `tests/chart/test_gateway.py`
- Modify: `values.yaml` (new `gateway` block), `templates/networkpolicy.yaml`, `templates/trino-configmap.yaml`, `templates/validate.yaml`, `templates/endpoints-configmap.yaml`

**Interfaces:**
- Produces these values:
  ```
  gateway:
    enabled: false
    gatewayClassName: ""
    parentRef: {name: "", namespace: "", sectionName: ""}
    domain: localhost
    hostTemplate: "{{ .route }}.{{ .release }}.{{ .domain }}"
    allowFromNamespaces: []
    extraRoutes: []        # {name, service, port, longLived}
  ```
  The template renders a `Gateway` named `<fullname>-gateway` only when `gatewayClassName` is set. Otherwise routes attach to `parentRef`, and `validate.yaml` fails if both or neither are set. Built-in routes, each only when its component is enabled: `jupyter` (8888, longLived), `trino` (8080), `airflow` (8080), `spark-history` (18080), `spark` (4040), `phoenix` (6006), `filer` (seaweedfs 8888). It also produces the endpoints key `GATEWAY_URL_<ROUTE>` = `http://<host>`.

- [ ] **Step 1: Write the failing tests.** In `test_gateway.py`:
  - `test_off_by_default`: no `Gateway` or `HTTPRoute`.
  - `test_namespaced_gateway`: with `gatewayClassName: datalab`, there is exactly one `Gateway` and no `GatewayClass`.
  - `test_hostnames_include_release`: release `datalab`, domain `localhost` → the jupyter route host is `jupyter.datalab.localhost`. Two renders with releases `a` and `b` share no hostname.
  - `test_longlived_jupyter`: the jupyter route has `timeouts.request == "0s"`.
  - `test_netpol_allows_gateway_ns`: `allowFromNamespaces: [envoy-gateway-system]` adds an ingress `namespaceSelector` with `kubernetes.io/metadata.name: envoy-gateway-system`.
  - `test_trino_forwarded`: the Trino `config.properties` contains `http-server.process-forwarded=true` only when the gateway is enabled.
  - `test_parentref_xor_class`: setting both fails the render.

  All tests use `api_versions=["gateway.networking.k8s.io/v1/Gateway"]`.
- [ ] **Step 2: Run the tests and confirm they fail.** Run `pytest tests/chart/test_gateway.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement the template, values (with `# --` comments), validation, NetworkPolicy and Trino change.** Fail the render with the hint "install a Gateway API implementation (hack/workstation/setup-minikube.sh)" when the API is missing.
- [ ] **Step 4: Run the tests, the golden test and the render policy.** Run `pytest tests/chart -q && helm template r charts/data-lab -n ns --api-versions gateway.networking.k8s.io/v1/Gateway --set gateway.enabled=true,gateway.gatewayClassName=x | python3 hack/check-chart-render.py r-data-lab ns`. Expected: PASS, with no cluster-scoped kinds.
- [ ] **Step 5: Commit.** Message: `chart: optional Gateway API routes (namespaced Gateway or shared parentRef)`.

### Task 3: Ollama GPU via DRA, tool-calling model

**Files:**
- Modify: `values.yaml` (`ollama.gpu.dra`), `templates/ollama.yaml`
- Test: `tests/chart/test_ollama.py`

**Interfaces:**
- Produces `ollama.gpu.dra: {enabled: false, deviceClassName: gpu.nvidia.com}`. When it is enabled (it requires `ollama.gpu.enabled`), the chart renders a namespaced `ResourceClaimTemplate` `<fullname>-ollama-gpu` (`resource.k8s.io/v1`) and the pod's `resourceClaims`. In that case the `nvidia.com/gpu` limit is not set.

- [ ] **Step 1: Write the failing tests.**
  - `test_device_plugin_limit`: with `gpu.enabled`, the limit `nvidia.com/gpu: 1` is set.
  - `test_dra_claim`: with `gpu.dra.enabled`, there is one ResourceClaimTemplate with `deviceClassName: gpu.nvidia.com`, the pod claims it, and the limit is absent.
  - `test_dra_requires_api`: without `resource.k8s.io/v1/ResourceClaimTemplate` in the API versions, the render fails.
- [ ] **Step 2: Confirm they fail, implement, confirm they pass.** Run `pytest tests/chart/test_ollama.py -q`. Expected: FAIL, then PASS after implementing.
- [ ] **Step 3: Commit.** Message: `chart: Ollama GPU through DRA (optional)`.

### Task 4: Spark Operator integration and the lakehouse e2e helm test

**Files:**
- Create: `charts/data-lab/files/e2e_lakehouse.py` (copy of `tests/e2e/e2e_lakehouse.py`), `charts/data-lab/templates/tests/test-lakehouse-e2e.yaml`, `hack/check-compose-parity.py`, `tests/chart/test_spark_apps.py`
- Modify: `values.yaml` (`spark.applications`, `tests.lakehouseE2e`), `templates/spark-rbac.yaml`, `templates/endpoints-configmap.yaml`

**Interfaces:**
- Produces `spark.applications: {enabled: false}`. When it is enabled, the chart renders a ServiceAccount `<fullname>-spark-apps` and a Role plus RoleBinding with the same rules as the Connect driver Role. It adds the endpoints keys `SPARK_APPLICATION_SERVICE_ACCOUNT` and `SPARK_DEFAULTS_CONFIGMAP` (the existing Spark conf ConfigMap name).
- Produces `tests.lakehouseE2e: {enabled: false}`, a helm test pod in the Spark image that runs `files/e2e_lakehouse.py` over Spark Connect (token) and Trino and exits 0 only on 19/19.
- Produces `hack/check-compose-parity.py` per D7, with exit code 1 and a diff on mismatch.

- [ ] **Step 1: Write the failing tests.**
  - `test_spark_apps_rbac`: the SA, Role and RoleBinding are present only when enabled, and the Role kind is never ClusterRole.
  - `test_lakehouse_test_pod`: the pod has the `helm.sh/hook: test` annotation, passes the PSS fields, and mounts the script from a ConfigMap whose data equals `tests/e2e/e2e_lakehouse.py`.
  - Parity: `python3 hack/check-compose-parity.py` exits 0 on the branch. A temporary edit of one Spark key in a copy of `spark/conf/spark-defaults.conf` makes it exit 1. Test this in `tests/unit/test_compose_parity.py` with a `tmp_path` copy.
- [ ] **Step 2: Confirm they fail, implement, confirm they pass.** Run `pytest tests/chart tests/unit -q`. Expected: FAIL, then PASS after implementing.
- [ ] **Step 3: Commit.** Message: `chart: SparkApplication service account, 19-check lakehouse helm test, compose parity check`.

### Task 5: Semantic Operator manager and server

**Files:**
- Create: `hack/crds/semantic.ossie.io_semanticmodels.yaml` (from `4fd7057`, with a header comment naming the source URL and commit), `charts/data-lab/templates/semantic.yaml`, `charts/data-lab/templates/semantic-configmap.yaml`, `charts/data-lab/templates/semantic-rbac.yaml`, `tests/chart/test_semantic.py`
- Modify: `values.yaml` (`semantic` block), `templates/_helpers.tpl` (`datalab.semanticServerName`), `templates/validate.yaml`, `templates/endpoints-configmap.yaml`, `templates/networkpolicy.yaml` (only if a separate rule is needed), `templates/NOTES.txt`

**Interfaces:**
- Consumes: the Trino Service name via `datalab.host (list . "trino")`.
- Produces these values:
  ```
  semantic:
    enabled: false
    manager: {image: {registry: public.ecr.aws, repository: data-on-eks/semantic-operator/manager, tag: v0.1.1, pullPolicy: IfNotPresent}, resyncPeriod: 30s, engineUser: semantic-manager, resources: {...}, <common pod keys>}
    server:  {image: {... repository: data-on-eks/semantic-operator/server ...}, replicas: 1, engineUser: semantic-server,
              auth: {mode: header, jwt: {jwksURL: "", issuer: "", audience: "", principalClaim: sub, roleClaim: role, groupsClaim: ""}},
              extraConfig: {}, resources: {...}, <common pod keys>}
    models: {dbtDemo: {enabled: true, catalog: iceberg, schema: dbt_demo}, extra: []}
  ```
  - Service `<fullname>-semantic:8090`.
  - Endpoints keys `SEMANTIC_MCP_URL`, `SEMANTIC_REST_URL`, `SEMANTIC_DEFAULT_ROLE` (D3).
  - Leader-election Lease name `<fullname>-semantic-manager`.

- [ ] **Step 1: Port the upstream config.** Read the upstream `configmap.yaml`, `manager-deployment.yaml`, `server-deployment.yaml` and `rbac.yaml` at `4fd7057`. Write down, as a comment block at the top of `semantic-configmap.yaml`, the config keys we set and the upstream defaults we keep.
- [ ] **Step 2: Write the failing tests.**
  - `test_off_by_default`: no `semantic` resources, and the golden test is still green.
  - `test_requires_crd`: enabled without `semantic.ossie.io/v1alpha1/SemanticModel` in the API versions fails with a message containing `hack/crds`.
  - `test_requires_trino`: `trino.enabled: false` fails.
  - `test_namespaced_only`: the rendered kinds include no ClusterRole, ClusterRoleBinding or CustomResourceDefinition.
  - `test_engine_points_at_release_trino`: the server config contains engine type `trino` and host `r-data-lab-trino`.
  - `test_header_auth_dev_only`: the default config has `mode: header` and `allowInsecureHeaderAuth: true`. With `server.auth.mode: jwt` and an empty `jwksURL` the render fails. With a `jwksURL`, `allowInsecureHeaderAuth` is absent or false.
  - `test_endpoints_keys`: `SEMANTIC_MCP_URL == "http://r-data-lab-semantic:8090/mcp"`.
  - `test_pss`: every semantic container passes the same restricted checks as `hack/check-chart-render.py`. Run that script on the render too.
- [ ] **Step 3: Write the one-release-per-namespace guard and its test.** In `validate.yaml`, when `semantic.enabled`, `lookup "apps/v1" "Deployment" .Release.Namespace ""` and fail if any Deployment has `app.kubernetes.io/component: semantic-manager` and an `app.kubernetes.io/instance` different from `.Release.Name`. The message names the other release and says "one semantic-enabled release per namespace (C14)". `test_semantic_second_release_rejected` can't run under `helm template`, because `lookup` is empty there, so it is covered in Task 6's kind e2e step. Here, add a unit test asserting that the `fail` text exists in the rendered template source.
- [ ] **Step 4: Confirm the tests fail, implement, confirm they pass.** Run `pytest tests/chart/test_semantic.py -q`. Expected: FAIL, then PASS after implementing.
- [ ] **Step 5: Commit.** Message: `chart: Kubeflow Semantic Operator (manager + server, namespaced) on the release's Trino`.

### Task 6: dbt-demo SemanticModel and the kind e2e `semantic` scenario

**Files:**
- Create: `charts/data-lab/templates/semantic-models.yaml`, `charts/data-lab/ci/semantic-values.yaml`, `tests/chart/test_semantic_model.py`
- Modify: `hack/chart-e2e-kind.sh` (scenario `semantic`), `hack/mcp-call.py` (header identity: `X-Semantic-User`, `X-Semantic-Role`), `.github/workflows/chart-e2e.yaml` (matrix entry `semantic`, CRD apply), `.github/workflows/chart-lint.yaml` (`--api-versions` for the new profiles)

**Interfaces:**
- Consumes: Task 5's values and Service.
- Produces: a `SemanticModel` named `<fullname>-dbt-demo`. `spec.ossie.name` is `dbt_demo`, and the model id that MCP `list_models` returns is recorded in the endpoints ConfigMap as `SEMANTIC_DEMO_MODEL`, so the notebook does not guess it. The metrics, dimensions, relationships and roles are exactly D3's.

- [ ] **Step 1: Write the failing tests.**
  - `test_model_rendered_when_enabled`: the model is present with `connection.catalog == "iceberg"` and `connection.database == "dbt_demo"`.
  - `test_metric_names`: the set equals `{total_revenue, order_count, units_sold, average_order_value, customer_count}`.
  - `test_revenue_excludes_returns`: the `total_revenue` ANSI expression contains `'returned'`.
  - `test_only_ansi_dialect`: every `dialect` equals `ANSI_SQL`.
  - `test_roles`: `marketing.denyFields` contains `email`.
  - `test_validates_against_crd`: `kubeconform -schema-location` with the vendored CRD converted to a JSON schema (`openapi2jsonschema.py`, as kubeconform documents) accepts the rendered model.
- [ ] **Step 2: Write the kind scenario.** In `hack/chart-e2e-kind.sh`, `semantic` does the following:
  1. `kubectl apply -f hack/crds/semantic.ossie.io_semanticmodels.yaml`.
  2. Install with `ci/semantic-values.yaml` (minimal plus `semantic.enabled`).
  3. **Apply-before-data:** wait until the model reports not-Ready.
  4. Seed `iceberg.dbt_demo.{fct_orders,stg_customers,dim_products}` with fixed rows through `trino --execute` in the Trino pod. 4 orders, one of them `returned`, so `total_revenue` has a known value `R`.
  5. Wait ≤ 120 s for the model to be Ready.
  6. Run `hack/mcp-call.py query_metric` for `total_revenue` and assert `== R`.
  7. Do the same with role `marketing` asking for `email` and assert a governance refusal.
  8. Install a second release with semantic enabled in the same namespace and assert `helm install` fails with "C14".
  9. Uninstall and assert the namespace is clean.

  The kubectl readiness condition comes from the CRD's status fields.
- [ ] **Step 3: Run the unit tests locally, then push and watch CI.** Run `pytest tests/chart -q`. Expected: PASS. Then CI `chart-e2e (semantic)` should be green.
- [ ] **Step 4: Commit.** Message: `chart: dbt-demo SemanticModel, kind e2e for the semantic layer`.

### Task 7: Chat helpers in `lab_utils.py`, Jupyter image dependencies

**Files:**
- Modify: `jupyter/notebooks/lab_utils.py`, `jupyter/Dockerfile` (pin `mcp==<exact latest 1.x>` and `ollama==<exact latest>` and record both in `docs/VERSIONS.md`)
- Create: `tests/unit/test_lab_utils_semantic.py`

**Interfaces:**
- Produces these functions in `lab_utils.py`:
  - `semantic_available() -> bool` (`SEMANTIC_MCP_URL` set)
  - `require_semantic() -> None`: raises `RuntimeError` with the text "The semantic layer runs only on the Helm chart with semantic.enabled=true (Docker Compose is frozen and does not include it)."
  - `mcp_tools(role: str = "analyst") -> list[dict]`: MCP `tools/list` turned into Ollama tool schemas `{"type": "function", "function": {name, description, parameters}}`
  - `mcp_call(name: str, args: dict, role: str = "analyst") -> dict`
  - `wait_for_model(model_id: str, timeout_s: int = 180) -> None`: polls `list_models` and raises `TimeoutError` naming the model
  - `ask_semantic(question: str, model: str = "qwen3:4b", role: str = "analyst", max_turns: int = 6) -> dict`: returns `{"answer", "tool_calls": list[str], "sql": str | None, "rows": list}`
  - `ask_raw_sql(question: str, tables: list[str], model: str = "qwen3:4b") -> dict`: returns `{"answer", "sql", "rows", "error": str | None}`
  - `require_tool_model(model: str) -> None`: raises when `"tools"` is not in the `ollama show` capabilities

  The headers are `X-Semantic-User: lab`, `X-Semantic-Role: <role>`.

- [ ] **Step 1: Write the failing tests** (no network, using `monkeypatch` or fakes):
  - `test_mcp_schema_to_ollama`: an MCP tool `{name: "query_metric", description: "d", inputSchema: {...}}` maps to the Ollama function schema with the same `parameters`.
  - `test_require_semantic_message`: the exact text above.
  - `test_wait_for_model_timeout`: a fake `list_models` that never contains the id raises `TimeoutError` within the timeout.
  - `test_ask_semantic_loop`: a fake chat that issues `list_metrics` then `query_metric` then a final answer returns `tool_calls == ["list_metrics", "query_metric"]` and the SQL from the `query_metric` result.
  - `test_ask_semantic_turn_cap`: a fake chat that never stops stops after `max_turns` and reports it in `answer`.
  - `test_require_tool_model`: a fake `show` without `tools` raises with the model name.
- [ ] **Step 2: Confirm they fail, implement, confirm they pass.** Run `pytest tests/unit/test_lab_utils_semantic.py -q`. Expected: FAIL, then PASS after implementing.
- [ ] **Step 3: Build the Jupyter image.** Run `docker build -t datalab-playground/jupyter ./jupyter`. Expected: success, and `python -c "import mcp, ollama"` in the image works.
- [ ] **Step 4: Commit.** Message: `jupyter: MCP and Ollama tool-calling helpers for the semantic layer`.

### Task 8: Notebook `semantic/01_chat_with_your_data.ipynb`

**Files:**
- Create: `jupyter/notebooks/semantic/01_chat_with_your_data.ipynb`
- Modify: `charts/data-lab/ci/workstation-values.yaml` (from Task 9; add `qwen3:4b` to `ollama.models`), or create it here if Task 9 has not run yet

**Interfaces:**
- Consumes: Task 7's helpers, Task 6's `SEMANTIC_DEMO_MODEL`.

- [ ] **Step 1: Write the notebook cells.** In order:
  1. A markdown intro: what a semantic layer is, and why the LLM never writes SQL.
  2. `require_semantic()`.
  3. `require_tool_model("qwen3:4b")`.
  4. `dbt build` if `fct_orders` is missing.
  5. `wait_for_model(os.environ["SEMANTIC_DEMO_MODEL"])`.
  6. List the metrics and dimensions.
  7. The 5 questions, each with its ground-truth SQL:
     - "What is total revenue by country?"
     - "How many orders per product category?"
     - "Units sold per product?"
     - "Average order value for each plan?"
     - "How many customers ordered in each country?"
  8. A loop: `ask_semantic`, `ask_raw_sql`, then compare both to the ground truth (numeric tolerance 0.01).
  9. Show a pandas score table (question, semantic ✓/✗, raw ✓/✗, tool calls, SQL).
  10. A governance demo: role `marketing` asking for emails is refused.
- [ ] **Step 2: Run it headless on the workstation.** Run `jupyter nbconvert --to notebook --execute jupyter/notebooks/semantic/01_chat_with_your_data.ipynb --ExecutePreprocessor.timeout=900` inside the Jupyter pod (Task 9's `e2e.sh` does this). Expected: no cell errors, and the semantic column scores 5/5. The raw column is informational and is not asserted.
- [ ] **Step 3: Check behaviour on Compose.** Run the first two cells on Compose. Expected: the `RuntimeError` message from Task 7 and nothing else.
- [ ] **Step 4: Commit.** Message: `notebooks: chat with your data through the Semantic Operator`.

### Task 9: Workstation tooling on the merged chart, remove `helm/`

**Files:**
- Create: `charts/data-lab/ci/workstation-values.yaml`, `hack/workstation/{setup-minikube.sh,build-images.sh,deploy.sh,e2e.sh,install-headlamp.sh,headlamp-token.sh,gateway-forward.sh,port-forward.sh,versions.env,headlamp-values.yaml,notebooks-pv.yaml,_common.sh}`
- Delete: `helm/` (whole tree)
- Modify: `.github/workflows/chart-lint.yaml` (lint `workstation-values.yaml` with `--api-versions` for Gateway, SemanticModel and ResourceClaimTemplate; run `hack/check-compose-parity.py` and `pytest tests/chart tests/unit`)

**Interfaces:**
- Consumes everything above. The profile sets:
  - jupyter, ollama (`gpu.enabled`, models `gemma3:4b`, `mxbai-embed-large:latest`, `qwen3:4b`), qdrant and phoenix on
  - `jupyter.persistence.existingClaim: datalab-notebooks`
  - `gateway.enabled` with `gatewayClassName: datalab`, `domain: localhost`, `allowFromNamespaces: [envoy-gateway-system]`, `extraRoutes: [{name: headlamp, service: headlamp, port: 80, longLived: true}]`
  - `spark.applications.enabled`, `semantic.enabled`, `tests.lakehouseE2e.enabled`
  - the Trino sizing from `helm/datalab/values-minikube.yaml`
- Script contracts:
  - `setup-minikube.sh [--gpu-mode device-plugin|dra|none]` keeps today's flags. It additionally creates the GatewayClass `datalab`, applies `hack/crds/semantic.ossie.io_semanticmodels.yaml` and `notebooks-pv.yaml` (a hostPath PV at `/datalab/notebooks`, storageClassName `datalab-notebooks`, plus a PVC `datalab-notebooks` in namespace `datalab`), and installs the Spark Operator with `spark.jobNamespaces={datalab}`.
  - `build-images.sh [--load]` builds hive-metastore, spark, jupyter and airflow-mcp, tagged `ghcr.io/tarekabouzeid/data-lab-playground/<name>:3.0.0` so the chart defaults resolve with `IfNotPresent`.
  - `deploy.sh` runs `helm upgrade --install datalab charts/data-lab -n datalab -f charts/data-lab/ci/workstation-values.yaml --wait`, then waits for the SemanticModel status, because `helm --wait` ignores CRs.
  - `e2e.sh` runs:
    1. `helm test`
    2. a SparkApplication running the lakehouse e2e with the `spark-apps` SA, applied only after Trino is Ready (the two-step order from `17e0f49`)
    3. all notebooks headless, including `semantic/01`
    4. a curl check of every gateway host
- [ ] **Step 1: Write the profile and confirm it lints and renders clean.** Run `helm lint --strict charts/data-lab -f charts/data-lab/ci/workstation-values.yaml --kube-version 1.34.0` and `helm template datalab charts/data-lab -n datalab -f …/workstation-values.yaml --api-versions gateway.networking.k8s.io/v1/Gateway --api-versions semantic.ossie.io/v1alpha1/SemanticModel | python3 hack/check-chart-render.py datalab-data-lab datalab`. Expected: PASS, with no cluster-scoped kinds and no hostPath.
- [ ] **Step 2: Port the scripts.** Use `git mv` for history, then make the edits per the contracts. Keep the verified fixes: the memory default (`630b587`), waiting for readiness and the two-step deploy (`e475ddf`, `17e0f49`), and Calico on WSL2 in `hack/kind-up.sh`. Run `bash -n` and `shellcheck` on every script. Expected: clean.
- [ ] **Step 3: Delete `helm/`.** Then `grep -rn "helm/datalab\|helm/scripts" --exclude-dir=.git .` must return only `docs/K8S_HELM_PLAN.md`, which gets a "Superseded by charts/data-lab (2026-10-30)" banner at the top.
- [ ] **Step 4: Commit.** Message: `workstation: minikube lab runs charts/data-lab; remove helm/datalab`.

### Task 10: Docs, chart metadata, version matrix

**Files:**
- Modify: `README.md` (one Kubernetes section; the two-chart table goes away; Compose marked frozen), `AGENTS.md` (pitfalls 14–19 merged and renumbered; new pitfalls for C13, C14, the CRD prerequisite, `ANSI_SQL`, `qwen3:4b`), `CLAUDE.md` (pitfalls 11–13 → one Helm pitfall; key-things and notebooks lists), `docs/VERSIONS.md` (§8 adds the Semantic Operator images, CRD commit, `qwen3:4b`, `mcp`/`ollama` pins; §9 becomes "Workstation tooling (`hack/workstation`)"), `docs/HELM_GAP_REPORT.md` (C13, C14, and verification rows for the new scenarios), `charts/data-lab/CHANGELOG.md` (0.2.0), `charts/data-lab/README.md.gotmpl` (workstation profile, semantic layer, Gateway, footprint rows), `.github/instructions/helm-chart.instructions.md`
- Generated: `charts/data-lab/values.schema.json`, `charts/data-lab/README.md`

- [ ] **Step 1: Update every file in the list.** Then run `python3 hack/gen-chart-schema.py --check` and `helm-docs -c charts -s file && git diff --exit-code charts/data-lab/README.md`. Expected: both clean after regenerating.
- [ ] **Step 2: Check for stale references.** Run `grep -rnE "helm/datalab|pitfall 1[4-9]|§9" --include=*.md .` and resolve each hit. Expected: none stale.
- [ ] **Step 3: Commit.** Message: `docs: unified chart, semantic layer, frozen Compose`.
- [ ] **Step 4: Ask the user** the RELEASE_NOTES question from Global Constraints, and write the entry only after a yes.

### Task 11: Local test runbook for the user (manual run on the workstation)

**Files:**
- Create: `docs/WORKSTATION_TESTING.md`

- [ ] **Step 1: Write the runbook.** It lists the commands and pass criteria in order:
  1. **Static checks:** `pytest tests/chart tests/unit -q`, `helm lint` on all profiles, `hack/check-compose-parity.py`.
  2. **Compose (frozen):** `tests/e2e/run-e2e.sh --build --down` gives 19/19, and `semantic/01` on Compose shows only the frozen-Compose message.
  3. **kind:** run `hack/kind-up.sh` once, then `hack/chart-e2e-kind.sh minimal|default|parallel|semantic`. All pass, and `minimal` is Ready in under 180 s.
  4. **Workstation:** run `setup-minikube.sh --gpu-mode device-plugin`, then `build-images.sh --load`, `deploy.sh`, `e2e.sh`. Then check by hand:
     - the URLs `jupyter|trino|airflow|spark-history|spark|phoenix|filer|headlamp.datalab.localhost:8080` load
     - `nvidia-smi` works in the Ollama pod
     - `semantic/01` scores 5/5 on the semantic path
     - the Headlamp sidebar shows Kubeflow → Spark
  5. **Sandpiper portability:** two releases in one namespace with defaults; `helm upgrade` from 0.1.0 to 0.2.0 with defaults creates and deletes nothing (`helm diff` or a `kubectl get all` before and after); and a second semantic release in the same namespace is rejected.

  Each step has a "paste this output back" line.
- [ ] **Step 2: Commit.** Message: `docs: workstation and Sandpiper test runbook`. Then push and confirm `chart-lint` and `chart-e2e` are green on the head commit.
