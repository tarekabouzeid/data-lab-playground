"""Helpers for render tests: run `helm template` on charts/data-lab and parse the output."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
CHART = REPO / "charts" / "data-lab"
KUBE_VERSION = "1.34.0"


class RenderError(Exception):
    """`helm template` failed; the message is helm's stderr."""


def _merge(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def render(values=None, profile=None, release="r", namespace="ns", api_versions=(), chart=CHART):
    """Render the chart and return the list of parsed documents. Raises RenderError on failure."""
    if shutil.which("helm") is None:
        pytest.skip("helm is not installed")
    cmd = ["helm", "template", release, str(chart), "-n", namespace, "--kube-version", KUBE_VERSION]
    if profile:
        cmd += ["-f", str(CHART / "ci" / profile)]
    for api in api_versions:
        cmd += ["--api-versions", api]
    cmd += ["-f", "-"]
    p = subprocess.run(cmd, input=json.dumps(values or {}), capture_output=True, text=True)
    if p.returncode != 0:
        raise RenderError(p.stderr)
    return [d for d in yaml.safe_load_all(p.stdout) if d]


def by_kind(docs, kind):
    return [d for d in docs if d.get("kind") == kind]


def labels(doc):
    return doc.get("metadata", {}).get("labels", {})


def workload(docs, component):
    """The Deployment/StatefulSet of a component (app.kubernetes.io/component)."""
    for d in docs:
        if d.get("kind") in ("Deployment", "StatefulSet") and labels(d).get("app.kubernetes.io/component") == component:
            return d
    raise KeyError(component)


def env_of(docs, component):
    """Env of the component's first container, by name."""
    c = workload(docs, component)["spec"]["template"]["spec"]["containers"][0]
    return {e["name"]: e for e in c.get("env", [])}


def configmap(docs, name):
    for d in by_kind(docs, "ConfigMap"):
        if d["metadata"]["name"] == name:
            return d["data"]
    raise KeyError(name)


@pytest.fixture
def merge():
    return _merge
