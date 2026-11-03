# Changelog

## [2.0.0 - 2026-11-03] (Snapmatic build)

Drop-in replacement for Visionatrix ComfyUI-Gemini 1.1.x. Same node id (`Ask_Gemini`),
same input names, same output; existing workflows run unchanged.

### Changed

- Moved from the legacy `google-generativeai` SDK (support ended 30 Nov 2025) to the
  current Google Gen AI SDK, `google-genai`.
- `model` now accepts any Gemini/Gemma model id. The dropdown is a shortlist only;
  ComfyUI no longer rejects a model because it is missing from the list.
- Model shortlist updated: gemini-3.8-flash, gemini-3.6-flash, gemini-3.5-flash-lite,
  gemini-3.1-flash-lite, gemini-3.1-pro-preview, gemini-3-flash-preview, gemini-3.5-flash,
  gemini-2.5-flash, gemini-2.5-flash-lite, gemini-2.5-pro, gemma-3-12b-it, gemma-3-27b-it.
- An empty, blocked or truncated-to-nothing reply now counts as a failure, so
  `error_fallback_value` is used instead of an empty string going downstream.
- Thought parts from thinking models are dropped from the output text.
- Failures are logged at WARNING with the reason (were silent at DEBUG).
- Gemma models: `system_instruction` is folded into the prompt (the API rejects it for Gemma).
- `temperature` max raised from 1 to 2 (Gemini's range).

### Added

- `thinking_level` (model default / minimal / low / medium / high).
- `timeout_seconds` (default 60) per request.
- `retries` (default 1) on 408/429/500/502/503/504.
- `safety_settings` options `BLOCK_LOW_AND_ABOVE` and `OFF`.

### Unchanged on purpose

- `seed` is still not sent to Gemini; it only forces the node to re-run. Sending it would
  make fixed-seed workflows return the same text every time.
- Empty `api_key` still means "use GOOGLE_API_KEY (or GEMINI_API_KEY) from the environment".
- `error_fallback_value` semantics: empty string = fail the job; linked input = evaluated
  only when Gemini fails.

## Earlier versions

See https://github.com/Visionatrix/ComfyUI-Gemini/blob/main/CHANGELOG.md
