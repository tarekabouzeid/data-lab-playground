"""Runs inside the Airflow api-server container (piped via kubectl exec -i ... python -).

  trigger <dag_id> <task_id> <marker>   wait for the DAG, run it, wait for success, assert the task log has <marker>
  read <dag_id> <run_id> <task_id> <marker>   only assert the stored log of an earlier run has <marker>

Prints RUN_ID=<id> after a trigger. Uses the admin password from the SimpleAuthManager password file.
"""
import json
import sys
import time
import urllib.request

BASE = "http://localhost:8080"


def call(method, path, token=None, body=None):
    req = urllib.request.Request(BASE + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json", "Accept": "application/json"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def token():
    user, pw = next(iter(json.load(open("/opt/airflow/auth/passwords.json")).items()))
    return call("POST", "/auth/token", body={"username": user, "password": pw})["access_token"]


def assert_log(tok, dag, run, task, marker):
    for _ in range(30):  # log may lag the state change by a moment
        logs = call("GET", f"/api/v2/dags/{dag}/dagRuns/{run}/taskInstances/{task}/logs/1?full_content=true", tok)
        if marker in json.dumps(logs):
            print(f"log of {dag}/{run}/{task} contains {marker}")
            return
        time.sleep(2)
    sys.exit(f"FAIL: marker {marker} not in task log: {json.dumps(logs)[:800]}")


def main():
    mode, tok = sys.argv[1], token()
    if mode == "read":
        dag, run, task, marker = sys.argv[2:6]
        assert_log(tok, dag, run, task, marker)
        return
    dag, task, marker = sys.argv[2:5]
    for _ in range(90):  # dag-processor refresh interval is 30 s
        try:
            call("GET", f"/api/v2/dags/{dag}", tok)
            break
        except urllib.error.HTTPError:
            time.sleep(3)
    else:
        sys.exit(f"FAIL: DAG {dag} never appeared")
    call("PATCH", f"/api/v2/dags/{dag}", tok, {"is_paused": False})
    run = call("POST", f"/api/v2/dags/{dag}/dagRuns", tok, {"logical_date": None})["dag_run_id"]
    for _ in range(90):
        state = call("GET", f"/api/v2/dags/{dag}/dagRuns/{run}", tok)["state"]
        if state in ("success", "failed"):
            break
        time.sleep(3)
    if state != "success":
        sys.exit(f"FAIL: run {run} ended in state {state}")
    assert_log(tok, dag, run, task, marker)
    print(f"RUN_ID={run}")


if __name__ == "__main__":
    main()
