from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from batch_label.types import AppConfig


DEFAULT_EXTENSIONS = ("jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff")
DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_TIMEOUT = 300.0
DEFAULT_PROMPT_FORMAT = "auto"
PROMPT_FORMATS = {"auto", "text", "json"}


class BatchLabelError(Exception):
    """Raised for user-facing application errors."""


def parse_extensions(raw_value: str | None) -> tuple[str, ...]:
    if raw_value is None:
        return DEFAULT_EXTENSIONS

    extensions = tuple(
        part.strip().lower().lstrip(".")
        for part in raw_value.split(",")
        if part.strip()
    )
    if not extensions:
        raise BatchLabelError("At least one file extension must be provided.")
    return extensions


def build_config(args: Namespace) -> AppConfig:
    images_path = Path(args.images).expanduser()
    prompt_file = Path(args.prompt_file).expanduser()
    base_url = args.base_url.rstrip("/")
    prompt_format = args.prompt_format

    if not images_path.exists():
        raise BatchLabelError(f"Images path does not exist: {images_path}")
    if not prompt_file.is_file():
        raise BatchLabelError(f"Prompt file does not exist: {prompt_file}")
    if not base_url:
        raise BatchLabelError("Base URL must not be empty.")
    if prompt_format not in PROMPT_FORMATS:
        raise BatchLabelError(f"Unsupported prompt format: {prompt_format}")
    if args.timeout <= 0:
        raise BatchLabelError("Timeout must be greater than zero.")

    return AppConfig(
        images_path=images_path,
        prompt_file=prompt_file,
        prompt_format=prompt_format,
        base_url=base_url,
        model=args.model,
        timeout=float(args.timeout),
        force=bool(args.force),
        recursive=bool(args.recursive),
        extensions=parse_extensions(args.extensions),
        verbose=bool(args.verbose),
    )


def _load_json_prompt(prompt_file: Path) -> str:
    try:
        payload = json.loads(prompt_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BatchLabelError(f"Prompt JSON file is invalid: {prompt_file}") from exc
    except OSError as exc:
        raise BatchLabelError(f"Failed to read prompt file {prompt_file}: {exc}") from exc

    if not isinstance(payload, dict):
        raise BatchLabelError(f"Prompt JSON file must contain an object: {prompt_file}")

    caption = payload.get("caption")
    if not isinstance(caption, str):
        raise BatchLabelError(
            f"Prompt JSON file must contain a string 'caption' key: {prompt_file}"
        )
    return caption


def load_prompt(prompt_file: Path, prompt_format: str = DEFAULT_PROMPT_FORMAT) -> str:
    if prompt_format == "json":
        return _load_json_prompt(prompt_file)
    if prompt_format == "auto" and prompt_file.suffix.lower() == ".json":
        return _load_json_prompt(prompt_file)

    try:
        return prompt_file.read_text(encoding="utf-8")
    except OSError as exc:
        raise BatchLabelError(f"Failed to read prompt file {prompt_file}: {exc}") from exc


def load_image_caption(image_path: Path) -> str | None:
    caption_path = image_path.with_name(f"{image_path.name}.json")
    if not caption_path.is_file():
        return None
    return _load_json_prompt(caption_path)
