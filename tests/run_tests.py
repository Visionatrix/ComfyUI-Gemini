"""End-to-end tests: real ComfyUI + new Ask_Gemini node + mock Gemini REST API."""

import json
import sys
import time
import urllib.error
import urllib.request
import uuid

BASE = "http://127.0.0.1:8188"
LOG = sys.argv[1]
OK_TEXT = "#NoonShare\n#Lets neighbours trade spare midday sun.\n#Short test description.\n#G01\n#Vertical 4:5 image test prompt."
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post(path, data):
    req = urllib.request.Request(BASE + path, json.dumps(data).encode(), {"content-type": "application/json"})
    try:
        return 200, json.load(opener.open(req))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def get(path):
    return json.load(opener.open(BASE + path))


def requests_log():
    try:
        return [json.loads(l) for l in open(LOG)]
    except FileNotFoundError:
        return []


def gemini(model, **over):
    """Ask_Gemini inputs shaped like the Prompt the Future workflow (no new inputs)."""
    inputs = {
        "seed": ["seed", 0],
        "model": model,
        "proxy": "",
        "prompt": ["prompt", 0],
        "api_key": "",
        "num_predict": 0,
        "temperature": 1,
        "response_type": "text",
        "safety_settings": "BLOCK_NONE",
        "system_instruction": "You are the idea generator. Reply in five # lines.",
        "error_fallback_value": "#NoonCharge\n#FALLBACK",
    }
    inputs.update(over)
    return {
        "seed": {"class_type": "PrimitiveInt", "inputs": {"value": int(time.time() * 1000) % 999999999}},
        "tmpl": {
            "class_type": "PrimitiveStringMultiline",
            "inputs": {"value": "R1: my building\nRANDOM NUMBER: (SEED)"},
        },
        "prompt": {"class_type": "StringReplace", "inputs": {"string": ["tmpl", 0], "find": "(SEED)", "replace": "42"}},
        "gem": {"class_type": "Ask_Gemini", "inputs": inputs},
        "out": {"class_type": "TestTextSink", "inputs": {"text": ["gem", 0]}},
    }


def run(graph, timeout=60):
    n_before = len(requests_log())
    t0 = time.time()
    code, r = post("/prompt", {"prompt": graph, "client_id": str(uuid.uuid4())})
    if code != 200:
        return {"rejected": r, "requests": [], "secs": 0}
    pid = r["prompt_id"]
    while time.time() - t0 < timeout:
        h = get(f"/history/{pid}")
        if pid in h and h[pid].get("status", {}).get("completed") is not None:
            st = h[pid]["status"]
            if st.get("status_str") == "success" or st.get("completed"):
                out = h[pid]["outputs"].get("out", {}).get("text", [None])[0]
                return {
                    "text": out,
                    "status": st["status_str"],
                    "requests": requests_log()[n_before:],
                    "secs": time.time() - t0,
                }
            return {
                "text": None,
                "status": st["status_str"],
                "messages": st.get("messages"),
                "requests": requests_log()[n_before:],
                "secs": time.time() - t0,
            }
        time.sleep(0.3)
    return {"timeout": True}


results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))


# 1. Production-shaped payload, current model, env key
r = run(gemini("gemini-3.6-flash"))
req = r["requests"][0]["body"] if r.get("requests") else {}
check("1 prod-shaped payload returns Gemini text (thought part dropped)", r.get("text") == OK_TEXT, r)
check(
    "1 one request, to gemini-3.6-flash",
    len(r["requests"]) == 1 and "/models/gemini-3.6-flash:generateContent" in r["requests"][0]["path"],
    r.get("requests"),
)
check("1 uses GOOGLE_API_KEY from env", r["requests"][0]["key"] == "env-test-key")
check("1 system instruction sent", "idea generator" in json.dumps(req.get("systemInstruction")))
check(
    "1 safety BLOCK_NONE on 4 categories",
    sorted(s["threshold"] for s in req.get("safetySettings", [])) == ["BLOCK_NONE"] * 4,
    req.get("safetySettings"),
)
gc = req.get("generationConfig", {})
check(
    "1 temperature=1, text/plain, no max tokens, no thinking config",
    gc.get("temperature") == 1
    and gc.get("responseMimeType") == "text/plain"
    and "maxOutputTokens" not in gc
    and "thinkingConfig" not in gc,
    gc,
)
check("1 seed substituted into prompt", "RANDOM NUMBER: 42" in json.dumps(req.get("contents")))

# 2. Model not in the dropdown list is accepted (this is what broke gemini-3.8-flash before)
r = run(gemini("gemini-4-argon"))
check(
    "2 unlisted model accepted and called", r.get("text") == OK_TEXT and "gemini-4-argon" in r["requests"][0]["path"], r
)

# 3. Garbage model id rejected at validation
r = run(gemini("bad model!"))
check(
    "3 invalid model id rejected by validation",
    "rejected" in r and "Invalid Gemini model id" in json.dumps(r["rejected"]),
    r,
)

# 4. Transient 503 retried once (retries default = 1)
r = run(gemini("mock-503-once"))
check("4 503 then 200 -> retried, real answer", r.get("text") == OK_TEXT and len(r["requests"]) == 2, r)

# 5-7. Failures fall back
for name, model in [("5 HTTP 500", "mock-500"), ("6 safety block", "mock-blocked"), ("7 empty reply", "mock-empty")]:
    r = run(gemini(model))
    check(f"{name} -> error_fallback_value", r.get("text") == "#NoonCharge\n#FALLBACK", r)

# 8. Timeout
r = run(gemini("mock-slow", timeout_seconds=2, retries=0))
check("8 timeout 2s -> fallback in < 5s", r.get("text") == "#NoonCharge\n#FALLBACK" and r["secs"] < 5, r)

# 9. Empty fallback means fail loudly
r = run(gemini("mock-500", error_fallback_value=""))
check("9 empty fallback -> job errors", r.get("status") == "error", r)

# 10. Linked (lazy) fallback
g = gemini("mock-500", error_fallback_value=["fb", 0])
g["fb"] = {"class_type": "PrimitiveStringMultiline", "inputs": {"value": "LINKED-FALLBACK"}}
r = run(g)
check("10 linked fallback evaluated lazily and used", r.get("text") == "LINKED-FALLBACK", r)

g = gemini("gemini-3.8-flash", error_fallback_value=["fb", 0])
g["fb"] = {"class_type": "PrimitiveStringMultiline", "inputs": {"value": "LINKED-FALLBACK"}}
r = run(g)
check("10b linked fallback not used on success", r.get("text") == OK_TEXT, r)

# 11. New inputs + per-node key + json + max tokens
r = run(
    gemini(
        "gemini-3.8-flash",
        api_key="node-key",
        thinking_level="low",
        num_predict=500,
        response_type="json",
        temperature=-0.05,
    )
)
req = r["requests"][0] if r.get("requests") else {}
gc = req.get("body", {}).get("generationConfig", {})
check("11 node api_key overrides env", req.get("key") == "node-key", req.get("key"))
check(
    "11 thinking level LOW (SDK sends snake_case key, same as calling the SDK directly), maxOutputTokens 500, json, temperature unset",
    (gc.get("thinkingConfig", {}).get("thinkingLevel") or gc.get("thinkingConfig", {}).get("thinking_level")) == "LOW"
    and gc.get("maxOutputTokens") == 500
    and gc.get("responseMimeType") == "application/json"
    and "temperature" not in gc,
    gc,
)

# 12. Gemma: system instruction folded into prompt
r = run(gemini("gemma-3-27b-it"))
body = r["requests"][0]["body"] if r.get("requests") else {}
check(
    "12 gemma: no systemInstruction field, folded into prompt",
    "systemInstruction" not in body and "idea generator" in json.dumps(body.get("contents")),
    body,
)

# 13. Image input
g = gemini("gemini-3.8-flash", image_1=["img", 0])
g["img"] = {"class_type": "EmptyImage", "inputs": {"width": 64, "height": 64, "batch_size": 1, "color": 0}}
r = run(g)
body = r["requests"][0]["body"] if r.get("requests") else {}
check(
    "13 image sent as inline PNG",
    (
        '"mimeType": "image/png"' in json.dumps(body.get("contents"))
        or '"mime_type": "image/png"' in json.dumps(body.get("contents"))
    ),
    json.dumps(body)[:300],
)

passed = sum(ok for _, ok in results)
print(f"\n{passed}/{len(results)} passed")
sys.exit(0 if passed == len(results) else 1)
