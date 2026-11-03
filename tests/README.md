End-to-end tests run against a real ComfyUI with a mock Gemini REST server (no API key needed).

1. `python3 tests/mock_gemini.py 9911 /tmp/requests.jsonl &`
2. Start ComfyUI with `GOOGLE_GEMINI_BASE_URL=http://127.0.0.1:9911/ GOOGLE_API_KEY=env-test-key NO_PROXY=127.0.0.1`
   and tests/test_text_sink.py copied into custom_nodes (test-only output node).
3. `python3 tests/run_tests.py /tmp/requests.jsonl`

last_run.txt is the result from 3 Nov 2026 (21/21).
