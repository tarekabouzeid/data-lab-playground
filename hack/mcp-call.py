"""Minimal MCP streamable-HTTP client (stdlib only), used by the kind e2e from inside the cluster.

usage: python mcp-call.py <url> <tool> [json-arguments]
Extra request headers (e.g. the semantic layer's X-Semantic-User / X-Semantic-Role) come from the
MCP_HEADERS environment variable as a JSON object.
Initializes a session, calls one tool and prints the tool's text result. Exit code 0 when the call
returned a result (a tool-level error is still printed, so the caller can assert on it), 1 on protocol errors.
"""
import json
import os
import sys
import urllib.request

URL = sys.argv[1]
HEADERS = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
           **json.loads(os.environ.get("MCP_HEADERS") or "{}")}
session = None


def rpc(method, params=None, rid=None):
    global session
    body = {"jsonrpc": "2.0", "method": method, "params": params or {}}
    if rid is not None:
        body["id"] = rid
    headers = dict(HEADERS, **({"mcp-session-id": session} if session else {}))
    with urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), headers), timeout=120) as r:
        session = r.headers.get("mcp-session-id") or session
        raw = r.read().decode()
    if rid is None or not raw.strip():
        return None
    if raw.lstrip().startswith("{"):
        return json.loads(raw)
    events = [line[5:].strip() for line in raw.splitlines() if line.startswith("data:")]
    return json.loads(events[-1])


def main():
    tool = sys.argv[2]
    args = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                       "clientInfo": {"name": "data-lab-e2e", "version": "1"}}, rid=1)
    rpc("notifications/initialized")
    res = rpc("tools/call", {"name": tool, "arguments": args}, rid=2)
    if "error" in res:
        print(json.dumps(res["error"]))
        sys.exit(1)
    for item in res["result"].get("content", []):
        print(item.get("text", ""))


if __name__ == "__main__":
    main()
