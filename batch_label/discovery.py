from __future__ import annotations

from pathlib import Path


def discover_images(
    images_path: Path,
    extensions: tuple[str, ...],
    recursive: bool,
) -> list[Path]:
    normalized_extensions = {f".{extension.lower()}" for extension in extensions}

    if images_path.is_file():
        return [images_path] if images_path.suffix.lower() in normalized_extensions else []

    pattern = "**/*" if recursive else "*"
    image_paths = [
        path
        for path in images_path.glob(pattern)
        if path.is_file() and path.suffix.lower() in normalized_extensions
    ]
    return sorted(image_paths)


def captions_output_dir(images_path: Path) -> Path:
    return images_path if images_path.is_dir() else images_path.parent


def sidecar_path_for(image_path: Path, output_dir: Path) -> Path:
    try:
        relative_parent = image_path.parent.relative_to(output_dir)
    except ValueError:
        relative_parent = Path()

    if relative_parent.parts:
        prefix = "__".join(relative_parent.parts)
        sidecar_name = f"{prefix}__{image_path.stem}.txt"
    else:
        sidecar_name = f"{image_path.stem}.txt"

    return output_dir / sidecar_name


def image_instruction_path_for(image_path: Path) -> Path:
    return image_path.with_name(f"{image_path.name}.json")
