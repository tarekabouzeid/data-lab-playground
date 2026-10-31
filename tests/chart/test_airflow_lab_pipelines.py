"""airflow.labPipelines: DAG tasks get the lab's client settings and, with the semantic layer, read-only SemanticModels."""
from conftest import by_kind, render, workload

API = ["semantic.ossie.io/v1alpha1/SemanticModel"]
ON = {"airflow": {"labPipelines": {"enabled": True}}}


def scheduler_env(docs):
    pod = workload(docs, "airflow")["spec"]["template"]["spec"]
    sched = next(c for c in pod["containers"] if c["name"] == "scheduler")
    return pod, {e["name"]: e for e in sched["env"]}


def test_off_by_default():
    docs = render()
    pod, env = scheduler_env(docs)
    assert pod["automountServiceAccountToken"] is False and "serviceAccountName" not in pod
    assert not {"SPARK_REMOTE", "TRINO_HOST", "SEMANTIC_MODEL_RESOURCE"} & set(env)
    assert not [d for d in by_kind(docs, "Role") if d["metadata"]["name"] == "r-data-lab-airflow"]


def test_client_env_without_semantic():
    pod, env = scheduler_env(render(ON))
    assert env["SPARK_REMOTE"]["value"].startswith("sc://r-data-lab-spark-connect")
    assert env["TRINO_HOST"]["value"].startswith("r-data-lab-trino") and env["TRINO_PORT"]["value"] == "8080"
    assert "SEMANTIC_MODEL_RESOURCE" not in env
    assert pod["automountServiceAccountToken"] is False       # no API access needed without the semantic layer


def test_spark_token_goes_into_the_remote_url():
    _, env = scheduler_env(render({**ON, "spark": {"auth": {"enabled": True}}}))
    assert env["SPARK_CONNECT_TOKEN"]["valueFrom"]["secretKeyRef"]["name"] == "r-data-lab-spark-connect"
    assert env["SPARK_REMOTE"]["value"].endswith(";authorization=Bearer $(SPARK_CONNECT_TOKEN)")


def test_semantic_read_only_access():
    docs = render({**ON, "semantic": {"enabled": True}}, api_versions=API)
    pod, env = scheduler_env(docs)
    assert env["SEMANTIC_MODEL_RESOURCE"]["value"] == "r-data-lab-dbt-demo"
    assert pod["serviceAccountName"] == "r-data-lab-airflow" and pod["automountServiceAccountToken"] is True
    role = next(d for d in by_kind(docs, "Role") if d["metadata"]["name"] == "r-data-lab-airflow")
    assert role["rules"] == [{"apiGroups": ["semantic.ossie.io"], "resources": ["semanticmodels"],
                              "verbs": ["get", "list", "watch"]}]
    binding = next(d for d in by_kind(docs, "RoleBinding") if d["metadata"]["name"] == "r-data-lab-airflow")
    assert binding["subjects"] == [{"kind": "ServiceAccount", "name": "r-data-lab-airflow", "namespace": "ns"}]
    assert not by_kind(docs, "ClusterRole") and not by_kind(docs, "ClusterRoleBinding")


def test_workstation_profile_runs_the_lab_pipelines():
    docs = render(profile="workstation-values.yaml", api_versions=API + ["gateway.networking.k8s.io/v1/Gateway",
                                                                         "gateway.networking.k8s.io/v1/HTTPRoute"])
    pod, env = scheduler_env(docs)
    assert pod["containers"][0]["image"].endswith("data-lab-playground/airflow:3.0.0")
    assert env["AIRFLOW__CORE__DAGS_FOLDER"]["value"] == "/opt/airflow/dags/ops/dags"
    assert "SEMANTIC_MODEL_RESOURCE" in env
