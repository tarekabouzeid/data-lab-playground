#!/usr/bin/env python3
"""Generate charts/data-lab/values.schema.json from values.yaml.

Types come from the defaults, descriptions from the helm-docs `# --` comments.
Every object is closed (additionalProperties: false) except the Kubernetes
pass-through fields listed in FREE_FORM, so a typo in a key fails `helm install`.

Usage: hack/gen-chart-schema.py [--check]
"""
import json
import pathlib
import re
import sys

import yaml

CHART = pathlib.Path(__file__).resolve().parent.parent / "charts" / "data-lab"

# Kubernetes / free-form objects passed through as-is (key name -> schema).
STR_MAP = {"type": "object", "additionalProperties": {"type": ["string", "number", "boolean"]}}
FREE_FORM = {
    "resources": {"type": "object", "description": "Kubernetes ResourceRequirements."},
    "affinity": {"type": "object", "description": "Kubernetes Affinity."},
    "podSecurityContext": {"type": "object", "description": "Kubernetes PodSecurityContext."},
    "securityContext": {"type": "object", "description": "Kubernetes SecurityContext."},
    "containerSecurityContext": {"type": "object", "description": "Kubernetes SecurityContext."},
    "nodeSelector": {"type": "object", "additionalProperties": {"type": "string"}},
    "podLabels": {"type": "object", "additionalProperties": {"type": "string"}},
    "podAnnotations": {"type": "object", "additionalProperties": {"type": "string"}},
    "commonLabels": {"type": "object", "additionalProperties": {"type": "string"}},
    "tolerations": {"type": "array", "items": {"type": "object"}},
    "extraEnv": {"type": "array", "items": {"type": "object", "required": ["name"]}},
    "extraIngressFrom": {"type": "array", "items": {"type": "object"}},
    "imagePullSecrets": {"type": "array", "items": {"type": "string", "minLength": 1}},
    "accessModes": {"type": "array", "minItems": 1, "items": {
        "type": "string", "enum": ["ReadWriteOnce", "ReadOnlyMany", "ReadWriteMany", "ReadWriteOncePod"]}},
    "extraConf": STR_MAP,
    "extraConfig": STR_MAP,
    "extraProperties": STR_MAP,
    "additionalCatalogs": {"type": "object", "additionalProperties": {"type": "string"}},
}

# Free-form fields by dotted path (take precedence over FREE_FORM's key names).
FREE_FORM_PATHS = {
    "ollama.models": {"type": "array", "items": {"type": "string", "minLength": 1}},
    "semantic.server.extraConfig": {"type": "object", "description": "Merged into the server config.yaml."},
    "semantic.models.extra": {"type": "array", "items": {"type": "object", "required": ["name", "spec"]}},
}

# Extra constraints by dotted path.
OVERRIDES = {
    "spark.master": {"enum": ["kubernetes", "local"]},
    "airflow.dags.source": {"enum": ["pvc", "gitSync", "none"]},
    "s3.endpoint": {"pattern": r"^(|https?://.+)$"},
    "s3.bucket": {"pattern": r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$"},
    "fullnameOverride": {"maxLength": 40},
    "spark.executor.requestCores": {"type": ["string", "number"]},
    "spark.executor.limitCores": {"type": ["string", "number"]},
    "spark.executor.instances": {"minimum": 0},
    "spark.executor.cores": {"minimum": 1},
    "ollama.gpu.count": {"minimum": 1},
    "semantic.server.auth.mode": {"enum": ["header", "jwt"]},
    "semantic.server.replicas": {"minimum": 1},
    "airflow.dags.gitSync.depth": {"minimum": 0},
    "seaweedfs.volumeMax": {"minimum": 1},
}
PULL_POLICY = {"enum": ["Always", "IfNotPresent", "Never"]}


def descriptions(text):
    """Map dotted key path -> helm-docs description."""
    out, stack, pending = {}, [], []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# --"):
            pending = [stripped[4:].strip()]
            continue
        if pending and stripped.startswith("#") and not stripped.startswith("# --"):
            pending.append(stripped[1:].strip())
            continue
        m = re.match(r"^(\s*)([A-Za-z0-9_]+):", line)
        if not m:
            if stripped and not stripped.startswith("#"):
                pending = []
            continue
        indent = len(m.group(1))
        while stack and stack[-1][0] >= indent:
            stack.pop()
        stack.append((indent, m.group(2)))
        if pending:
            out[".".join(k for _, k in stack)] = " ".join(pending)
            pending = []
    return out


def schema_for(value, path, docs):
    key = path.rsplit(".", 1)[-1]
    if path in FREE_FORM_PATHS:
        s = dict(FREE_FORM_PATHS[path])
    elif key in FREE_FORM:
        s = dict(FREE_FORM[key])
    elif isinstance(value, dict):
        s = {"type": "object", "additionalProperties": False,
             "properties": {k: schema_for(v, f"{path}.{k}" if path else k, docs) for k, v in value.items()}}
    elif isinstance(value, bool):
        s = {"type": "boolean"}
    elif isinstance(value, int):
        s = {"type": "integer"}
    elif isinstance(value, float):
        s = {"type": "number"}
    elif isinstance(value, list):
        s = {"type": "array"}
    else:
        s = {"type": "string"}
    if key == "pullPolicy":
        s.update(PULL_POLICY)
    if key == "tag" and path.endswith("image.tag"):
        s["minLength"] = 1
    s.update(OVERRIDES.get(path, {}))
    if path in docs:
        s["description"] = docs[path]
    return s


def main():
    text = (CHART / "values.yaml").read_text()
    schema = {"$schema": "https://json-schema.org/draft-07/schema#", "title": "data-lab values"}
    schema.update(schema_for(yaml.safe_load(text), "", descriptions(text)))
    rendered = json.dumps(schema, indent=2) + "\n"
    target = CHART / "values.schema.json"
    if "--check" in sys.argv:
        if not target.exists() or target.read_text() != rendered:
            sys.exit("values.schema.json is out of date: run hack/gen-chart-schema.py")
        print("values.schema.json is up to date")
    else:
        target.write_text(rendered)
        print(f"wrote {target}")


if __name__ == "__main__":
    main()
