#!/usr/bin/env python3
"""Keep charts/data-lab aligned with the Docker Compose stack.

Fails (exit 1) when
  * a key of spark/conf/spark-defaults.conf is missing from, or differs in, the chart's rendered
    spark-defaults.conf (the Spark Connect ConfigMap), unless ALLOWED below says why it may differ;
  * charts/data-lab/files/e2e_lakehouse.py is not identical to tests/e2e/e2e_lakehouse.py.

Usage: hack/check-compose-parity.py [--compose-conf PATH] [--e2e PATH]   (needs helm)
"""
import argparse
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CHART = REPO / "charts" / "data-lab"

# Compose keys the chart deliberately sets differently (or not at all), with the reason.
ALLOWED = {
    "spark.master": "the chart runs executors on Kubernetes (k8s://...)",
    "spark.hadoop.fs.s3a.endpoint": "the release's SeaweedFS Service (or s3.endpoint)",
    "spark.hadoop.fs.s3a.access.key": "credential: injected from the S3 Secret at start-up",
    "spark.hadoop.fs.s3a.secret.key": "credential: injected from the S3 Secret at start-up",
    "spark.executor.memory": "sized by spark.executor.memory in values.yaml",
    "spark.executor.cores": "sized by spark.executor.cores in values.yaml",
    "spark.sql.adaptive.coalescePartitions.enabled": "Spark's default is already true",
}
# S3A timeouts: Compose pins them; the chart keeps Hadoop 3.4.2's defaults (its kind e2e passes with them).
ALLOWED_PREFIXES = {
    "spark.hadoop.fs.s3a.connection.": "S3A timeouts: Hadoop 3.4.2 defaults in the chart",
    "spark.hadoop.fs.s3a.socket.timeout": "S3A timeouts: Hadoop 3.4.2 defaults in the chart",
    "spark.hadoop.fs.s3a.request.timeout": "S3A timeouts: Hadoop 3.4.2 defaults in the chart",
    "spark.hadoop.fs.s3a.threads.keepalivetime": "S3A timeouts: Hadoop 3.4.2 defaults in the chart",
    "spark.hadoop.fs.s3a.multipart.purge.age": "S3A timeouts: Hadoop 3.4.2 defaults in the chart",
}


def parse_conf(text):
    """spark-defaults.conf -> {key: value}; the last occurrence wins, as in Spark."""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        first = line.split()[0]
        sep = "=" if "=" in first else None  # Compose writes key=value, the chart key<spaces>value
        key, value = line.split(sep, 1) if sep else (first, line[len(first):])
        out[key.strip()] = value.strip()
    return out


def chart_conf():
    out = subprocess.run(["helm", "template", "r", str(CHART), "-n", "ns", "--kube-version", "1.34.0",
                          "-s", "templates/spark-configmap.yaml"], check=True, capture_output=True, text=True).stdout
    for doc in yaml.safe_load_all(out):
        if doc and doc.get("kind") == "ConfigMap" and "spark-defaults.conf" in doc.get("data", {}):
            return parse_conf(doc["data"]["spark-defaults.conf"])
    raise SystemExit("no spark-defaults.conf in the chart's Spark ConfigMap")


def allowed(key):
    return key in ALLOWED or any(key.startswith(p) for p in ALLOWED_PREFIXES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compose-conf", default=str(REPO / "spark" / "conf" / "spark-defaults.conf"))
    ap.add_argument("--e2e", default=str(REPO / "tests" / "e2e" / "e2e_lakehouse.py"))
    args = ap.parse_args()

    problems = []
    compose, chart = parse_conf(Path(args.compose_conf).read_text()), chart_conf()
    for key, value in sorted(compose.items()):
        if allowed(key):
            continue
        if key not in chart:
            problems.append(f"{key}: set in Compose ({value}) but missing from the chart")
        elif chart[key] != value:
            problems.append(f"{key}: Compose {value!r} != chart {chart[key]!r}")
    if Path(args.e2e).read_text() != (CHART / "files" / "e2e_lakehouse.py").read_text():
        problems.append("charts/data-lab/files/e2e_lakehouse.py differs from tests/e2e/e2e_lakehouse.py (copy it)")

    for p in problems:
        print("FAIL", p)
    print(f"compose parity: {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
