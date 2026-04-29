from __future__ import annotations

from typing import Any

import httpx

from batch_label.client import LlamaServerClient, LlamaServerError, resolve_model_id
from batch_label.config import DEFAULT_BASE_URL, DEFAULT_TIMEOUT
from batch_label.image_prep import USER_IMAGE_INSTRUCTION, prepare_image_tensor
from batch_label.messages import append_prompt_text, format_batch_text_output
from batch_label.presets import get_prompt_preset_text, list_prompt_presets


COMFYUI_FREE_URL = "http://127.0.0.1:8188/free"


def unload_comfyui_models(timeout: float) -> None:
    try:
        response = httpx.post(
            COMFYUI_FREE_URL,
            json={"unload_models": True, "free_memory": False},
            timeout=httpx.Timeout(timeout=timeout, connect=5.0),
        )
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Failed to call ComfyUI /free: {exc}") from exc

    if response.status_code >= 400:
        raise RuntimeError(f"ComfyUI /free returned HTTP {response.status_code}: {response.text.strip()}")


def _iter_batch_images(image: Any) -> list[Any]:
    shape = getattr(image, "shape", None)
    if shape is None:
        return [image]
    if len(shape) == 4:
        return [image[index] for index in range(int(shape[0]))]
    if len(shape) == 3:
        return [image]
    raise ValueError(f"Unsupported IMAGE tensor shape: {shape}")


class BatchLabelLlamaVisionText:
    CATEGORY = "BatchLabel/Llama.cpp"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, Any]]:
        return {
            "required": {
                "image": ("IMAGE", {}),
                "system_prompt_preset": (list_prompt_presets(), {"default": "short_caption"}),
                "instructions": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Additional user instructions for this image...",
                    },
                ),
            },
            "optional": {
                "system_prompt_custom": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Optional extra system prompt text...",
                    },
                ),
                "base_url": ("STRING", {"default": DEFAULT_BASE_URL}),
                "model": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Leave empty to auto-select when exactly one model is exposed",
                    },
                ),
                "timeout": ("FLOAT", {"default": DEFAULT_TIMEOUT, "min": 1.0, "max": 3600.0, "step": 1.0}),
            },
        }

    def run(
        self,
        image: Any,
        system_prompt_preset: str,
        instructions: str,
        system_prompt_custom: str = "",
        base_url: str = DEFAULT_BASE_URL,
        model: str = "",
        timeout: float = DEFAULT_TIMEOUT,
    ) -> tuple[str]:
        system_prompt = append_prompt_text(
            get_prompt_preset_text(system_prompt_preset),
            system_prompt_custom,
        )
        user_text = append_prompt_text(USER_IMAGE_INSTRUCTION, instructions)
        outputs: list[str] = []
        resolved_model_id: str | None = None
        main_error: Exception | None = None

        try:
            with LlamaServerClient(base_url, timeout) as client:
                resolved_model_id = resolve_model_id(client, model.strip() or None)
                for batch_index, image_tensor in enumerate(_iter_batch_images(image), start=1):
                    try:
                        prepared_image = prepare_image_tensor(image_tensor)
                    except Exception as exc:  # pragma: no cover - exact error text varies by tensor type
                        raise RuntimeError(f"Failed to prepare image {batch_index}: {exc}") from exc

                    try:
                        label_response = client.label_image(
                            resolved_model_id,
                            system_prompt,
                            user_text,
                            prepared_image,
                        )
                    except LlamaServerError as exc:
                        raise RuntimeError(f"Failed to label image {batch_index}: {exc}") from exc

                    outputs.append(label_response.text)

                client.maybe_unload_model(resolved_model_id)
        except Exception as exc:
            main_error = exc

        cleanup_error: Exception | None = None
        try:
            unload_comfyui_models(timeout)
        except Exception as exc:
            cleanup_error = exc

        if main_error is not None:
            if cleanup_error is not None:
                raise RuntimeError(f"{main_error} Cleanup also failed: {cleanup_error}") from main_error
            raise main_error

        if cleanup_error is not None:
            raise cleanup_error

        return (format_batch_text_output(outputs),)


NODE_CLASS_MAPPINGS = {
    "BatchLabelLlamaVisionText": BatchLabelLlamaVisionText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchLabelLlamaVisionText": "BatchLabel Llama.cpp Vision Text",
}
