#!/usr/bin/env bash
# Fails when the chart drifts from docker-compose.yaml / the Spark config files:
#  1. every image reference in values.yaml == the compose images (and vice versa)
#  2. spark.sparkConf in values.yaml contains every key of spark/conf/spark-defaults.conf (except spark.master), same values
#  3. spark/conf/spark-defaults.conf == jupyter/spark-defaults.conf   (CLAUDE.md pitfall 2)
#  4. helm/datalab/files/e2e_lakehouse.py == tests/e2e/e2e_lakehouse.py
#  5. helm/headlamp-values.yaml pins (image tag, pluginctl, Kubeflow plugin) == helm/scripts/versions.env
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require python3
python3 - "$ROOT" <<'PY'
import re, sys, pathlib
root = pathlib.Path(sys.argv[1]); bad = []

# 1. images ------------------------------------------------------------------------------------------------
compose = {m.group(1).strip() for l in (root/"docker-compose.yaml").read_text().splitlines()
           if (m := re.match(r"^\s*image:\s*([^\s#]+)", l))}
vals = (root/"helm/datalab/values.yaml").read_text()
chart = {f"{r}:{t}" for r, t in re.findall(r'repository:\s*([^\s,}]+),\s*tag:\s*"?([^"\s}]+)"?', vals)}
for img in sorted(compose - chart): bad.append(f"image in docker-compose.yaml but not in values.yaml: {img}")
for img in sorted(chart - compose): bad.append(f"image in values.yaml but not in docker-compose.yaml: {img}")

# 2. spark conf --------------------------------------------------------------------------------------------
def parse_defaults(p):
    d = {}
    for l in p.read_text().splitlines():
        l = l.strip()
        if l and not l.startswith("#") and "=" in l:
            k, v = l.split("=", 1); d[k.strip()] = v.strip()      # last one wins, like Spark
    return d
defaults = parse_defaults(root/"spark/conf/spark-defaults.conf"); defaults.pop("spark.master", None)
block = vals.split("sparkConf:", 1)[1]
chart_conf = {}
for l in block.splitlines():
    m = re.match(r'^    ([A-Za-z0-9_.\-]+):\s*"(.*)"\s*$', l)
    if m: chart_conf[m.group(1)] = m.group(2)
    elif l.strip() and not l.startswith("    ") and not l.startswith("#"): break   # next top-level key
for k, v in defaults.items():
    if k not in chart_conf: bad.append(f"spark-defaults.conf key missing in values.yaml spark.sparkConf: {k}")
    elif chart_conf[k] != v: bad.append(f"spark conf differs for {k}: compose={v!r} chart={chart_conf[k]!r}")
if "spark.master" in chart_conf: bad.append("spark.sparkConf must not set spark.master (the operator owns it)")

# 3 + 4. copies --------------------------------------------------------------------------------------------
if (root/"spark/conf/spark-defaults.conf").read_bytes() != (root/"jupyter/spark-defaults.conf").read_bytes():
    bad.append("spark/conf/spark-defaults.conf and jupyter/spark-defaults.conf differ")
if (root/"helm/datalab/files/e2e_lakehouse.py").read_bytes() != (root/"tests/e2e/e2e_lakehouse.py").read_bytes():
    bad.append("helm/datalab/files/e2e_lakehouse.py differs from tests/e2e/e2e_lakehouse.py (copy it again)")

# 5. Headlamp pins -------------------------------------------------------------------------------------------
ver = dict(re.findall(r"^([A-Z_]+)=([^\s#]+)", (root/"helm/scripts/versions.env").read_text(), re.M))
hv = (root/"helm/headlamp-values.yaml").read_text()
def need(label, pattern, expected):
    m = re.search(pattern, hv, re.M)
    if not m: bad.append(f"headlamp-values.yaml: cannot find {label}")
    elif m.group(1) != expected: bad.append(f"headlamp-values.yaml {label} is {m.group(1)!r}, versions.env says {expected!r}")
need("image.tag", r"^  tag:\s*(\S+)", "v" + ver["HEADLAMP_VERSION"])
need("pluginsManager.version", r'^  version:\s*"([^"]+)"', ver["HEADLAMP_PLUGINCTL_VERSION"])
need("Kubeflow plugin version", r"^        version:\s*(\S+)", ver["HEADLAMP_KUBEFLOW_PLUGIN_VERSION"])

if bad:
    print("PARITY FAILED:"); [print("  -", b) for b in bad]; sys.exit(1)
print(f"parity ok: {len(compose)} images, {len(defaults)} spark conf keys, Headlamp pins")
PY
