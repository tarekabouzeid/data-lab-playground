from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_notebook_copy_of_the_dbt_demo_model_matches_the_chart():
    a = (REPO / "jupyter/notebooks/semantic/dbt_demo_model.yaml").read_text()
    b = (REPO / "charts/data-lab/files/semantic/dbt-demo.yaml").read_text()
    assert a == b, "copy charts/data-lab/files/semantic/dbt-demo.yaml to jupyter/notebooks/semantic/dbt_demo_model.yaml"
