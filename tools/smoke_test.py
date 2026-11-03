"""Smoke-test the deployed Ask_Gemini node through a running ComfyUI (stdlib only).

    python3 tools/smoke_test.py                          # http://127.0.0.1:8188, gemini-3.8-flash
    python3 tools/smoke_test.py http://127.0.0.1:8188 gemini-3.6-flash

Uses the server's GOOGLE_API_KEY, like production workflows. The fallback value is
empty, so a Gemini failure shows up as an error here instead of being hidden.
"""

import json
import random
import sys
import time
import urllib.error
import urllib.request
import uuid

base = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8188").rstrip("/")
model = sys.argv[2] if len(sys.argv) > 2 else "gemini-3.8-flash"
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

graph = {
    "1": {
        "class_type": "Ask_Gemini",
        "inputs": {
            "prompt": "Reply with exactly: #OK#<one word for the future of energy>",
            "model": model,
            "safety_settings": "BLOCK_NONE",
            "response_type": "text",
            "api_key": "",
            "proxy": "",
            "system_instruction": "Follow the user's format exactly.",
            "error_fallback_value": "",
            "seed": random.randint(0, 2**31),
            "temperature": 1,
            "num_predict": 0,
        },
    },
    "2": {"class_type": "PreviewAny", "inputs": {"source": ["1", 0]}},
}

req = urllib.request.Request(
    f"{base}/prompt",
    json.dumps({"prompt": graph, "client_id": str(uuid.uuid4())}).encode(),
    {"content-type": "application/json"},
)
try:
    prompt_id = json.load(opener.open(req))["prompt_id"]
except urllib.error.HTTPError as e:
    body = e.read().decode()
    print("Rejected by ComfyUI:", body)
    if "PreviewAny" in body:
        print("This ComfyUI has no PreviewAny node; test through a Snapmatic test job instead.")
    sys.exit(1)

t0 = time.time()
while time.time() - t0 < 120:
    hist = json.load(opener.open(f"{base}/history/{prompt_id}"))
    if prompt_id in hist and hist[prompt_id].get("status", {}).get("completed") is not None:
        entry = hist[prompt_id]
        if entry["status"].get("status_str") == "success":
            print(f"OK in {time.time() - t0:.1f}s ({model}):", entry["outputs"]["2"]["text"][0])
            sys.exit(0)
        for kind, msg in entry["status"].get("messages", []):
            if kind == "execution_error":
                print("FAILED:", msg.get("exception_type"), msg.get("exception_message"))
        sys.exit(1)
    time.sleep(0.5)
print("Timed out waiting for ComfyUI")
sys.exit(1)
