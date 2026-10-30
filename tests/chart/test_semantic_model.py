"""The dbt-demo SemanticModel: certified metrics over the dbt demo tables, valid against the vendored CRD."""
import jsonschema
import pytest
import yaml

from conftest import REPO, by_kind, configmap, render

API = ["semantic.ossie.io/v1alpha1/SemanticModel"]
ON = {"semantic": {"enabled": True}}


def model(docs, name="r-data-lab-dbt-demo"):
    return next(m for m in by_kind(docs, "SemanticModel") if m["metadata"]["name"] == name)


def crd_schema():
    class Loader(yaml.SafeLoader):
        pass
    Loader.add_constructor("tag:yaml.org,2002:value", lambda loader, node: loader.construct_scalar(node))
    crd = yaml.load((REPO / "hack" / "crds" / "semantic.ossie.io_semanticmodels.yaml").read_text(), Loader=Loader)
    return next(v for v in crd["spec"]["versions"] if v["name"] == "v1alpha1")["schema"]["openAPIV3Schema"]


def test_model_rendered_when_enabled():
    m = model(render(ON, api_versions=API))
    assert m["apiVersion"] == "semantic.ossie.io/v1alpha1"
    assert m["spec"]["connection"]["catalog"] == "iceberg"
    assert m["spec"]["connection"]["database"] == "dbt_demo"
    assert m["spec"]["ossie"]["name"] == "dbt_demo"


def test_model_off_when_disabled():
    v = {"semantic": {"enabled": True, "models": {"dbtDemo": {"enabled": False}}}}
    assert not by_kind(render(v, api_versions=API), "SemanticModel")


def test_metric_names():
    names = {x["name"] for x in model(render(ON, api_versions=API))["spec"]["ossie"]["metrics"]}
    assert names == {"total_revenue", "order_count", "units_sold", "average_order_value", "customer_count"}


def test_revenue_excludes_returns():
    metrics = {x["name"]: x for x in model(render(ON, api_versions=API))["spec"]["ossie"]["metrics"]}
    assert "'returned'" in metrics["total_revenue"]["expression"]["dialects"][0]["expression"]


def test_only_ansi_dialect():
    spec = model(render(ON, api_versions=API))["spec"]["ossie"]
    exprs = [x["expression"] for x in spec["metrics"]] + [f["expression"] for d in spec["datasets"] for f in d["fields"]]
    assert {d["dialect"] for e in exprs for d in e["dialects"]} == {"ANSI_SQL"}


def test_dimensions():
    spec = model(render(ON, api_versions=API))["spec"]["ossie"]
    dims = {f"{d['name']}.{f['name']}" for d in spec["datasets"] for f in d["fields"] if "dimension" in f}
    assert dims == {"stg_customers.country", "stg_customers.plan", "dim_products.category",
                    "dim_products.product_name", "fct_orders.order_date", "fct_orders.status"}


def test_roles():
    gov = model(render(ON, api_versions=API))["spec"]["governance"]
    roles = {r["name"]: r for r in gov["roles"]}
    assert gov["defaultRole"] == "analyst"
    assert roles["marketing"]["denyFields"] == ["stg_customers.email"]
    assert "denyFields" not in roles["analyst"]


def test_validates_against_crd():
    m = model(render(ON, api_versions=API))
    jsonschema.validate({k: m[k] for k in ("apiVersion", "kind", "spec")}, crd_schema())


def test_extra_models():
    extra = {"name": "mine", "spec": {"connection": {"catalog": "iceberg", "database": "x"}, "ossie": {"name": "mine"}}}
    docs = render({"semantic": {"enabled": True, "models": {"extra": [extra]}}}, api_versions=API)
    assert model(docs, "r-data-lab-mine")["spec"]["ossie"]["name"] == "mine"


def test_demo_model_endpoint_key():
    assert configmap(render(ON, api_versions=API), "r-data-lab-endpoints")["SEMANTIC_DEMO_MODEL"] == "dbt_demo"


def test_semantic_profile_lints():
    docs = render(profile="semantic-values.yaml", api_versions=API)
    assert by_kind(docs, "SemanticModel")


def hook_docs(docs):
    return [d for d in docs if d["metadata"].get("annotations", {}).get("helm.sh/hook") == "pre-delete"]


def test_uninstall_hook_deletes_models_first():
    docs = hook_docs(render(ON, api_versions=API))
    kinds = {d["kind"]: d for d in docs}
    assert set(kinds) == {"ServiceAccount", "Role", "RoleBinding", "Job"}
    assert kinds["Role"]["rules"] == [{"apiGroups": ["semantic.ossie.io"], "resources": ["semanticmodels"],
                                       "verbs": ["get", "list", "delete", "patch"]}]
    job = kinds["Job"]["spec"]["template"]["spec"]
    assert job["serviceAccountName"] == "r-data-lab-semantic-cleanup"
    env = {e["name"]: e.get("value") for e in job["containers"][0]["env"]}
    assert env["SELECTOR"] == "app.kubernetes.io/instance=r"
    for d in docs:
        assert "hook-succeeded" in d["metadata"]["annotations"]["helm.sh/hook-delete-policy"]


def test_no_hook_without_semantic():
    assert not hook_docs(render(api_versions=API))
