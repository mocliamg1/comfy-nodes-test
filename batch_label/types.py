from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    images_path: Path
    prompt_file: Path
    prompt_format: str
    base_url: str
    model: str | None
    timeout: float
    force: bool
    recursive: bool
    extensions: tuple[str, ...]
    verbose: bool


@dataclass(frozen=True)
class ImageJob:
    image_path: Path
    sidecar_path: Path


@dataclass(frozen=True)
class PreparedImage:
    mime_type: str
    data_uri: str
    width: int
    height: int


@dataclass(frozen=True)
class ServerModelInfo:
    id: str


@dataclass(frozen=True)
class LabelResponse:
    text: str
    raw_status: int
    model_id: str


@dataclass(frozen=True)
class BatchResult:
    processed: int
    skipped: int
    failed: int
