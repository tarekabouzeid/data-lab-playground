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


MODEL_FILE = REPO / "charts" / "data-lab" / "files" / "semantic" / "dbt-demo.yaml"


def test_spec_is_the_model_file():
    spec = model(render(ON, api_versions=API))["spec"]
    assert spec == yaml.safe_load(MODEL_FILE.read_text())


def test_connection_from_values():
    v = {"semantic": {"enabled": True, "models": {"dbtDemo": {"catalog": "lake", "schema": "shop"}}}}
    conn = model(render(v, api_versions=API))["spec"]["connection"]
    assert conn == {"catalog": "lake", "database": "shop", "viewDatabase": "lake.semantic_views"}


def test_metric_names():
    names = {x["name"] for x in model(render(ON, api_versions=API))["spec"]["ossie"]["metrics"]}
    assert names == {"total_revenue", "order_count", "units_sold", "average_order_value", "customer_count",
                     "gross_margin", "refunded_amount", "refund_rate", "amount_collected", "payment_count"}


def test_seven_datasets_joined_into_one_graph():
    spec = model(render(ON, api_versions=API))["spec"]["ossie"]
    names = {d["name"] for d in spec["datasets"]}
    assert names == {"fct_orders", "fct_payments", "fct_refunds", "stg_customers", "dim_products",
                     "dim_product_costs", "dim_regions"}
    edges = {(r["from"], r["to"]) for r in spec["relationships"]}
    assert edges == {("fct_orders", "stg_customers"), ("fct_orders", "dim_products"), ("dim_products", "dim_product_costs"),
                     ("stg_customers", "dim_regions"), ("fct_payments", "fct_orders"), ("fct_refunds", "fct_orders")}
    for d in spec["datasets"]:
        assert d["primary_key"], f"{d['name']} needs a primary key (the planner dedupes fan-out on it)"


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
                    "dim_products.product_name", "fct_orders.order_date", "fct_orders.status", "fct_payments.method",
                    "fct_refunds.reason", "dim_product_costs.supplier", "dim_regions.sales_region"}


def test_roles():
    gov = model(render(ON, api_versions=API))["spec"]["governance"]
    roles = {r["name"]: r for r in gov["roles"]}
    assert gov["defaultRole"] == "analyst"
    assert roles["marketing"]["denyFields"] == ["stg_customers.email"]
    assert "denyFields" not in roles["analyst"]
    assert roles["dach_sales"]["rowFilters"] == [{"dataset": "dim_regions", "predicate": "sales_region = 'DACH'"}]


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


def test_uninstall_hook_tolerates_404():
    job = next(d for d in hook_docs(render(ON, api_versions=API)) if d["kind"] == "Job")
    script = job["spec"]["template"]["spec"]["containers"][0]["command"][2]
    assert "e.code == 404" in script
    compile(script, "hook", "exec")
