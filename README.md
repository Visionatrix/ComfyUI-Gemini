# ComfyUI-Gemini 2.0 (google-genai)

`Ask_Gemini` for ComfyUI, rebuilt on Google's current Gen AI SDK (`google-genai`).
It replaces the Visionatrix ComfyUI-Gemini 1.x node, which uses the legacy
`google-generativeai` SDK that Google stopped supporting on 30 Nov 2025.

**Drop-in:** same node id, input names and output. GEM, Prompt the Future and any other
workflow using `Ask_Gemini` runs without edits.

## What changes for us

- **Any model id works.** Put `gemini-3.8-flash` (or whatever Google ships next) in the
  workflow's `model` field and ComfyUI accepts it. The old node rejected anything not in
  its hard-coded list.
- **Empty/blocked replies use the fallback** instead of passing an empty string on.
- New optional inputs: `thinking_level`, `timeout_seconds` (60), `retries` (1, on
  408/429/5xx).
- Failures are logged at WARNING with the reason.

Full list in CHANGELOG.md.

## Install on a GPU server

```bash
unzip ComfyUI-Gemini-2.0.0.zip && cd ComfyUI-Gemini
./install.sh /path/to/ComfyUI /path/to/comfyui/venv/bin/python
# restart ComfyUI, then:
python3 tools/smoke_test.py http://127.0.0.1:8188 gemini-3.8-flash
```

The installer backs up the current node to `ComfyUI/custom_nodes_backup/`, shows what pip
would change, asks before installing, then copies the new node in.

Rollback: `./install.sh --rollback /path/to/ComfyUI` and restart ComfyUI.

The API key comes from `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) in the ComfyUI environment
when the node's `api_key` is empty, same as before.

## Inputs

| Input | Notes |
|---|---|
| prompt | User message. |
| model | Dropdown shortlist, but any Gemini/Gemma id is accepted. |
| safety_settings | One threshold for harassment, hate, sexual and dangerous content. |
| response_type | `text` or `json` (sets the response MIME type). |
| api_key, proxy | Empty = environment key / no proxy. |
| image_1..3 | Optional images sent with the prompt. |
| system_instruction | Folded into the prompt for Gemma models. |
| error_fallback_value | Returned if Gemini fails. Empty string = fail the job. A linked input is only evaluated on failure. |
| seed | Forces a re-run; not sent to Gemini. |
| temperature | -0.05 = model default. |
| num_predict | Max output tokens; 0 = model default. Includes thinking tokens on thinking models. |
| thinking_level | `model default`, `minimal`, `low`, `medium`, `high`. Gemini 3.x only; gemini-3.8-flash has no `minimal`. Gemini 2.5 models error on this (they use thinking budgets), so leave it on `model default` for them. |
| timeout_seconds | Per request; 0 = no timeout. |
| retries | Extra attempts on 408/429/500/502/503/504. |

Based on Visionatrix/ComfyUI-Gemini (MIT).
