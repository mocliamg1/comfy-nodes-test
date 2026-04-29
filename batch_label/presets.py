from __future__ import annotations

from importlib.resources import files


PROMPT_PRESET_IDS = (
    "short_caption",
    "detailed_description",
    "ocr_text_and_scene",
    "diffusion_json_blueprint",
    "nano_banana_json_blueprint",
)


def list_prompt_presets() -> list[str]:
    return list(PROMPT_PRESET_IDS)


def get_prompt_preset_text(preset_id: str) -> str:
    if preset_id not in PROMPT_PRESET_IDS:
        raise ValueError(f"Unknown prompt preset: {preset_id}")

    preset_path = files("batch_label").joinpath("prompt_presets", f"{preset_id}.txt")
    return preset_path.read_text(encoding="utf-8").strip()
