"""Tiny stand-in for the Gemini REST API, to test the node end to end without a key."""

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOG = sys.argv[2]
counts = {}
lock = threading.Lock()

OK_TEXT = "#NoonShare\n#Lets neighbours trade spare midday sun.\n#Short test description.\n#G01\n#Vertical 4:5 image test prompt."


def ok_body(text=OK_TEXT):
    return {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {"text": "internal reasoning", "thought": True},
                        {"text": text, "thoughtSignature": "c2ln"},
                    ],
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 20},
    }


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
        model = self.path.split("/models/")[1].split(":")[0]
        with lock:
            counts[model] = counts.get(model, 0) + 1
            n = counts[model]
            with open(LOG, "a") as f:
                f.write(
                    json.dumps({"path": self.path, "n": n, "key": self.headers.get("x-goog-api-key"), "body": body})
                    + "\n"
                )
        status, resp = 200, ok_body()
        if model == "mock-503-once" and n == 1:
            status, resp = 503, {"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}}
        elif model == "mock-500":
            status, resp = 500, {"error": {"code": 500, "message": "boom", "status": "INTERNAL"}}
        elif model == "mock-blocked":
            resp = {"promptFeedback": {"blockReason": "SAFETY"}}
        elif model == "mock-slow":
            time.sleep(5)
        elif model == "mock-empty":
            resp = {
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": "", "thoughtSignature": "c2ln"}]},
                        "finishReason": "MAX_TOKENS",
                    }
                ]
            }
        data = json.dumps(resp).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
