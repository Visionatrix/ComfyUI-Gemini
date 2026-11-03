"""Ask_Gemini on the current Google Gen AI SDK (google-genai).

Drop-in replacement for the Visionatrix ComfyUI-Gemini node, which was built on
the legacy google-generativeai SDK (support ended 30 Nov 2025). The node id
("Ask_Gemini"), every input name, the output and the error_fallback_value
behaviour are unchanged, so existing workflows run without edits.

What is new:
- model accepts any Gemini/Gemma model id. The dropdown is only a shortlist;
  ComfyUI no longer rejects a model just because it is missing from the list.
- optional thinking_level, timeout_seconds and retries inputs.
- an empty or blocked reply counts as a failure (so error_fallback_value is used)
  instead of passing an empty string downstream.
"""

import logging
import random
import re
import threading

from google import genai
from google.genai import types
from torch import Tensor

from .utils import images_to_pillow

LOGGER = logging.getLogger("ComfyUI-Gemini")

# Shortlist for the dropdown only. Any other model id typed into an API-format
# workflow is accepted too (see VALIDATE_INPUTS).
MODELS = [
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
    "gemini-3.5-flash",  # deprecated by Google, auto-routed to gemini-3.6-flash
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
    "gemma-3-12b-it",
    "gemma-3-27b-it",
]

SAFETY_SETTINGS = ["BLOCK_NONE", "BLOCK_ONLY_HIGH", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_LOW_AND_ABOVE", "OFF"]

# The legacy SDK applied a single threshold string to these four categories.
HARM_CATEGORIES = [
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
]

THINKING_LEVELS = ["model default", "minimal", "low", "medium", "high"]

RETRY_STATUS_CODES = [408, 429, 500, 502, 503, 504]

MODEL_ID_RE = re.compile(r"^(models/)?[A-Za-z0-9][A-Za-z0-9._:-]*$")

_clients: dict[tuple, genai.Client] = {}
_clients_lock = threading.Lock()


def _get_client(api_key: str | None, proxy: str | None, timeout_seconds: float, retries: int) -> genai.Client:
    """Return a cached client for this key/proxy/timeout/retry combination."""
    cache_key = (api_key or "", proxy or "", float(timeout_seconds), int(retries))
    with _clients_lock:
        client = _clients.get(cache_key)
        if client is None:
            http_options = types.HttpOptions(
                timeout=int(timeout_seconds * 1000) if timeout_seconds > 0 else None,
                retry_options=(
                    types.HttpRetryOptions(
                        attempts=retries + 1,
                        initial_delay=1.0,
                        max_delay=8.0,
                        http_status_codes=RETRY_STATUS_CODES,
                    )
                    if retries > 0
                    else None
                ),
                client_args={"proxy": proxy} if proxy else None,
            )
            # api_key=None makes the SDK read GOOGLE_API_KEY / GEMINI_API_KEY from the environment.
            client = genai.Client(api_key=api_key or None, http_options=http_options)
            _clients[cache_key] = client
        return client


def _response_text(response: types.GenerateContentResponse) -> str:
    """Join the answer's text parts (skipping thought parts); raise if there is no usable text."""
    candidates = response.candidates or []
    if candidates and candidates[0].content and candidates[0].content.parts:
        text = "".join(p.text for p in candidates[0].content.parts if p.text and not p.thought)
        if text.strip():
            return text
    block_reason = getattr(response.prompt_feedback, "block_reason", None) if response.prompt_feedback else None
    finish_reason = candidates[0].finish_reason if candidates else None
    raise RuntimeError(f"Gemini returned no text (finish_reason={finish_reason}, block_reason={block_reason})")


def generate_text(
    prompt: str,
    model: str,
    safety_settings: str = "BLOCK_NONE",
    response_type: str = "text",
    api_key: str | None = None,
    proxy: str | None = None,
    images: list | None = None,
    system_instruction: str | None = None,
    temperature: float | None = None,
    num_predict: int | None = None,
    thinking_level: str | None = None,
    timeout_seconds: float = 60.0,
    retries: int = 1,
) -> str:
    """One generate_content call with the legacy node's semantics. Raises on any failure."""
    model = (model or "").strip()
    if model.startswith("models/"):
        model = model[len("models/") :]
    system_instruction = system_instruction or None
    contents: list = [prompt, *(images or [])]

    # Gemma on the Gemini API rejects system instructions; fold them into the prompt instead.
    if system_instruction and model.startswith("gemma"):
        contents[0] = f"{system_instruction}\n\n{prompt}"
        system_instruction = None

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json" if response_type == "json" else "text/plain",
        safety_settings=[types.SafetySetting(category=c, threshold=safety_settings) for c in HARM_CATEGORIES],
    )
    if temperature is not None and temperature >= 0:
        config.temperature = temperature
    if num_predict is not None and num_predict > 0:
        config.max_output_tokens = num_predict
    if thinking_level and thinking_level != "model default":
        config.thinking_config = types.ThinkingConfig(thinking_level=thinking_level.upper())

    client = _get_client(api_key, proxy, timeout_seconds, retries)
    response = client.models.generate_content(model=model, contents=contents, config=config)
    return _response_text(response)


class GeminiNode:
    @classmethod
    def INPUT_TYPES(cls):  # noqa
        seed = random.randint(1, 2**31)

        return {
            "required": {
                "prompt": ("STRING", {"default": "Why number 42 is important?", "multiline": True}),
                "safety_settings": (SAFETY_SETTINGS,),
                "response_type": (["text", "json"],),
                "model": (MODELS,),
            },
            "optional": {
                "api_key": ("STRING", {}),
                "proxy": ("STRING", {}),
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "system_instruction": ("STRING", {}),
                "error_fallback_value": ("STRING", {"lazy": True}),
                "seed": ("INT", {"default": seed, "min": 0, "max": 2**31, "step": 1}),
                "temperature": ("FLOAT", {"default": -0.05, "min": -0.05, "max": 2, "step": 0.05}),
                "num_predict": ("INT", {"default": 0, "min": 0, "max": 1048576, "step": 1}),
                # New inputs go last so widget order in saved UI workflows is unchanged.
                "thinking_level": (THINKING_LEVELS, {"default": "model default"}),
                "timeout_seconds": ("FLOAT", {"default": 60.0, "min": 0, "max": 600, "step": 1}),
                "retries": ("INT", {"default": 1, "min": 0, "max": 5, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "ask_gemini"

    CATEGORY = "Gemini"

    @classmethod
    def VALIDATE_INPUTS(cls, model):  # noqa
        # Naming `model` here makes ComfyUI skip its "value not in list" check for it,
        # so new Gemini models work without editing MODELS.
        if not isinstance(model, str) or not MODEL_ID_RE.match(model.strip()):
            return f"Invalid Gemini model id: {model!r}"
        return True

    def __init__(self):
        self.text_output: str | None = None
        self.last_error: str | None = None

    def ask_gemini(self, **kwargs):
        if self.text_output is not None:
            return (self.text_output,)
        fallback = kwargs.get("error_fallback_value")
        if fallback is None:
            raise RuntimeError(f"Ask_Gemini failed and no error_fallback_value is set: {self.last_error}")
        return (fallback,)

    def check_lazy_status(
        self,
        prompt: str,
        safety_settings: str,
        response_type: str,
        model: str,
        api_key: str | None = None,
        proxy: str | None = None,
        image_1: Tensor | list[Tensor] | None = None,
        image_2: Tensor | list[Tensor] | None = None,
        image_3: Tensor | list[Tensor] | None = None,
        system_instruction: str | None = None,
        error_fallback_value: str | None = None,
        temperature: float | None = None,
        num_predict: int | None = None,
        thinking_level: str | None = None,
        timeout_seconds: float | None = None,
        retries: int | None = None,
        **kwargs,
    ):
        self.text_output = None
        self.last_error = None
        images = []
        for image in [image_1, image_2, image_3]:
            if image is not None:
                images.extend(images_to_pillow(image))
        try:
            self.text_output = generate_text(
                prompt=prompt,
                model=model,
                safety_settings=safety_settings,
                response_type=response_type,
                api_key=api_key,
                proxy=proxy,
                images=images,
                system_instruction=system_instruction,
                temperature=temperature,
                num_predict=num_predict,
                thinking_level=thinking_level,
                timeout_seconds=60.0 if timeout_seconds is None else timeout_seconds,
                retries=1 if retries is None else retries,
            )
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            if error_fallback_value is None:
                # Fallback is a linked (lazy) input that hasn't been evaluated yet: ask ComfyUI for it.
                LOGGER.warning("Ask_Gemini (%s) failed, requesting error_fallback_value: %s", model, self.last_error)
                return ["error_fallback_value"]
            if error_fallback_value == "":
                raise
            LOGGER.warning("Ask_Gemini (%s) failed, using error_fallback_value: %s", model, self.last_error)
        return []


NODE_CLASS_MAPPINGS = {
    "Ask_Gemini": GeminiNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Ask_Gemini": "Ask Gemini",
}
