import logging
import os
import random

from google import genai
from google.genai import types
from torch import Tensor

from .utils import images_to_pillow, temporary_env_var

SAFETY_CATEGORIES = [
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
    "HARM_CATEGORY_CIVIC_INTEGRITY",
]


class GeminiNode:

    @classmethod
    def INPUT_TYPES(cls):  # noqa
        seed = random.randint(1, 2**31)

        return {
            "required": {
                "prompt": ("STRING", {"default": "Why number 42 is important?", "multiline": True}),
                "safety_settings": (["BLOCK_NONE", "BLOCK_ONLY_HIGH", "BLOCK_MEDIUM_AND_ABOVE"],),
                "response_type": (["text", "json"],),
                "model": (
                    [
                        "gemma-3-12b-it",
                        "gemma-3-27b-it",
                        "gemini-3.1-flash-lite",
                        "gemini-3.5-flash",
                        "gemini-3.5-flash-lite",
                        "gemini-3.6-flash",
                        "gemini-3.7-flash",
                        "gemini-3.1-pro-preview",
                    ],
                ),
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
                "temperature": ("FLOAT", {"default": -0.05, "min": -0.05, "max": 1, "step": 0.05}),
                "num_predict": ("INT", {"default": 0, "min": 0, "max": 1048576, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "ask_gemini"

    CATEGORY = "Gemini"

    def __init__(self):
        self.text_output: str | None = None

    def ask_gemini(self, **kwargs):
        return (kwargs["error_fallback_value"] if self.text_output is None else self.text_output,)

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
        **kwargs,
    ):
        self.text_output = None
        if not system_instruction:
            system_instruction = None
        images_to_send = []
        for image in [image_1, image_2, image_3]:
            if image is not None:
                images_to_send.extend(images_to_pillow(image))
        client_kwargs = {}
        if api_key or "GOOGLE_API_KEY" not in os.environ:
            client_kwargs["api_key"] = api_key
        config_kwargs = {
            "response_mime_type": "application/json" if response_type == "json" else "text/plain",
            "safety_settings": [
                types.SafetySetting(category=cat, threshold=safety_settings) for cat in SAFETY_CATEGORIES
            ],
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if temperature is not None and temperature >= 0:
            config_kwargs["temperature"] = temperature
        if num_predict is not None and num_predict > 0:
            config_kwargs["max_output_tokens"] = num_predict
        try:
            with temporary_env_var("HTTP_PROXY", proxy), temporary_env_var("HTTPS_PROXY", proxy):
                client = genai.Client(**client_kwargs)
                response = client.models.generate_content(
                    model=model,
                    contents=[prompt, *images_to_send],
                    config=types.GenerateContentConfig(**config_kwargs),
                )
            self.text_output = response.text
        except Exception:
            if error_fallback_value is None:
                logging.getLogger("ComfyUI-Gemini").debug("ComfyUI-Gemini: exception occurred:", exc_info=True)
                return ["error_fallback_value"]
            if error_fallback_value == "":
                raise
        return []


NODE_CLASS_MAPPINGS = {
    "Ask_Gemini": GeminiNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Ask_Gemini": "Ask Gemini",
}
