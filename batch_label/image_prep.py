from __future__ import annotations

import base64
import io
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from batch_label.types import PreparedImage


MAX_PIXELS = 1_000_000
USER_IMAGE_INSTRUCTION = (
    "Label this image according to the system instructions. Return only the final label text."
)

try:
    RESAMPLING_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # pragma: no cover
    RESAMPLING_LANCZOS = Image.LANCZOS


def calculate_resize_dimensions(
    width: int,
    height: int,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    if width * height <= max_pixels:
        return width, height

    scale = math.sqrt(max_pixels / float(width * height))
    resized_width = max(1, int(width * scale))
    resized_height = max(1, int(height * scale))
    return resized_width, resized_height


def image_has_alpha(image: Image.Image) -> bool:
    if image.mode in {"RGBA", "LA"}:
        return True
    if image.mode == "P" and "transparency" in image.info:
        return True
    return False


def _finalize_prepared_image(image: Image.Image, max_pixels: int = MAX_PIXELS) -> PreparedImage:
    resized_width, resized_height = calculate_resize_dimensions(*image.size, max_pixels=max_pixels)
    if (resized_width, resized_height) != image.size:
        image = image.resize((resized_width, resized_height), RESAMPLING_LANCZOS)

    use_png = image_has_alpha(image)
    buffer = io.BytesIO()

    if use_png:
        if image.mode not in {"RGBA", "LA"}:
            image = image.convert("RGBA")
        image.save(buffer, format="PNG")
        mime_type = "image/png"
    else:
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(buffer, format="JPEG", quality=90)
        mime_type = "image/jpeg"

    encoded_image = base64.b64encode(buffer.getvalue()).decode("ascii")
    data_uri = f"data:{mime_type};base64,{encoded_image}"

    return PreparedImage(
        mime_type=mime_type,
        data_uri=data_uri,
        width=image.width,
        height=image.height,
    )


def _channel_to_byte(value: Any) -> int:
    if not isinstance(value, (int, float)):
        raise TypeError(f"Tensor pixel values must be numeric, got {type(value).__name__}")
    if 0.0 <= float(value) <= 1.0:
        return max(0, min(255, round(float(value) * 255.0)))
    return max(0, min(255, round(float(value))))


def _coerce_tensor_image_data(image_tensor: Any) -> tuple[int, int, int, list[list[list[Any]]]]:
    if hasattr(image_tensor, "detach"):
        image_tensor = image_tensor.detach()
    if hasattr(image_tensor, "cpu"):
        image_tensor = image_tensor.cpu()
    if hasattr(image_tensor, "tolist"):
        image_tensor = image_tensor.tolist()

    if not isinstance(image_tensor, list) or not image_tensor:
        raise TypeError("Expected a non-empty image tensor.")

    if not isinstance(image_tensor[0], list) or not image_tensor[0]:
        raise TypeError("Image tensor must be HxWxC.")

    if not isinstance(image_tensor[0][0], list):
        raise TypeError("Image tensor must be HxWxC.")

    height = len(image_tensor)
    width = len(image_tensor[0])
    channels = len(image_tensor[0][0])
    if channels not in {3, 4}:
        raise ValueError(f"Expected 3 or 4 channels, got {channels}")

    for row in image_tensor:
        if not isinstance(row, list) or len(row) != width:
            raise ValueError("Image tensor rows must have consistent width.")
        for pixel in row:
            if not isinstance(pixel, list) or len(pixel) != channels:
                raise ValueError("Image tensor pixels must have a consistent channel count.")

    return width, height, channels, image_tensor


def prepare_image_tensor(image_tensor: Any, max_pixels: int = MAX_PIXELS) -> PreparedImage:
    width, height, channels, image_data = _coerce_tensor_image_data(image_tensor)
    mode = "RGBA" if channels == 4 else "RGB"
    raw_bytes = bytearray()

    for row in image_data:
        for pixel in row:
            raw_bytes.extend(_channel_to_byte(channel) for channel in pixel)

    image = Image.frombytes(mode, (width, height), bytes(raw_bytes))
    return _finalize_prepared_image(image, max_pixels=max_pixels)


def prepare_image(image_path: Path, max_pixels: int = MAX_PIXELS) -> PreparedImage:
    with Image.open(image_path) as opened_image:
        image = ImageOps.exif_transpose(opened_image)
        return _finalize_prepared_image(image, max_pixels=max_pixels)
