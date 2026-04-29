from pathlib import Path

import pytest
from PIL import Image

from batch_label.client import build_chat_payload
from batch_label.config import BatchLabelError, load_image_caption, load_prompt
from batch_label.discovery import (
    captions_output_dir,
    discover_images,
    image_instruction_path_for,
    sidecar_path_for,
)
from batch_label.image_prep import (
    MAX_PIXELS,
    USER_IMAGE_INSTRUCTION,
    calculate_resize_dimensions,
    prepare_image,
    prepare_image_tensor,
)
from batch_label.messages import append_prompt_text, format_batch_text_output
from batch_label.presets import get_prompt_preset_text, list_prompt_presets
from batch_label.types import PreparedImage


class FakeTensor:
    def __init__(self, data):
        self._data = data
        self.shape = self._infer_shape(data)

    def __getitem__(self, index: int):
        return FakeTensor(self._data[index])

    def tolist(self):
        return self._data

    @staticmethod
    def _infer_shape(data) -> tuple[int, ...]:
        if not isinstance(data, list):
            return ()
        if not data:
            return (0,)
        return (len(data),) + FakeTensor._infer_shape(data[0])


def create_image(path: Path, size: tuple[int, int], mode: str = "RGB") -> None:
    color = (255, 0, 0, 255) if "A" in mode else (255, 0, 0)
    Image.new(mode, size, color=color).save(path)


def make_tensor_image(width: int, height: int, channels: int) -> FakeTensor:
    pixel = [0.2, 0.4, 0.6, 0.8][:channels]
    row = [list(pixel) for _ in range(width)]
    return FakeTensor([row[:] for _ in range(height)])


def test_discover_images_recursively(tmp_path: Path) -> None:
    create_image(tmp_path / "top.jpg", (10, 10))
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    create_image(nested_dir / "nested.png", (10, 10))

    paths = discover_images(tmp_path, ("jpg", "png"), recursive=True)

    assert [path.name for path in paths] == ["nested.png", "top.jpg"]


def test_discover_images_non_recursive(tmp_path: Path) -> None:
    create_image(tmp_path / "top.jpg", (10, 10))
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    create_image(nested_dir / "nested.png", (10, 10))

    paths = discover_images(tmp_path, ("jpg", "png"), recursive=False)

    assert [path.name for path in paths] == ["top.jpg"]


def test_sidecar_path_for_uses_output_directory() -> None:
    assert sidecar_path_for(
        Path("/tmp/example/nested/photo.jpeg"),
        Path("/tmp/example"),
    ) == Path("/tmp/example/nested__photo.txt")


def test_sidecar_path_for_keeps_top_level_names_simple() -> None:
    assert sidecar_path_for(
        Path("/tmp/example/photo.jpeg"),
        Path("/tmp/example"),
    ) == Path("/tmp/example/photo.txt")


def test_captions_output_dir_for_single_image_is_parent() -> None:
    assert captions_output_dir(Path("/tmp/example/photo.jpeg")) == Path("/tmp/example")


def test_image_instruction_path_for_appends_json_suffix() -> None:
    assert image_instruction_path_for(
        Path("/tmp/example/photo.png"),
    ) == Path("/tmp/example/photo.png.json")


@pytest.mark.parametrize(
    ("width", "height"),
    [
        (2000, 1500),
        (1200, 2500),
    ],
)
def test_calculate_resize_dimensions_caps_at_one_megapixel(width: int, height: int) -> None:
    resized_width, resized_height = calculate_resize_dimensions(width, height)

    assert resized_width * resized_height <= MAX_PIXELS
    assert resized_width < width
    assert resized_height < height


def test_calculate_resize_dimensions_keeps_small_images_unchanged() -> None:
    assert calculate_resize_dimensions(800, 600) == (800, 600)


def test_prepare_image_uses_png_when_alpha_is_present(tmp_path: Path) -> None:
    image_path = tmp_path / "alpha.png"
    create_image(image_path, (100, 100), mode="RGBA")

    prepared = prepare_image(image_path)

    assert prepared.mime_type == "image/png"
    assert prepared.data_uri.startswith("data:image/png;base64,")


def test_prepare_image_uses_jpeg_for_non_alpha_images(tmp_path: Path) -> None:
    image_path = tmp_path / "plain.png"
    create_image(image_path, (100, 100), mode="RGB")

    prepared = prepare_image(image_path)

    assert prepared.mime_type == "image/jpeg"
    assert prepared.data_uri.startswith("data:image/jpeg;base64,")


def test_prepare_image_tensor_uses_png_when_alpha_is_present() -> None:
    prepared = prepare_image_tensor(make_tensor_image(20, 10, 4))

    assert prepared.mime_type == "image/png"
    assert prepared.data_uri.startswith("data:image/png;base64,")


def test_prepare_image_tensor_uses_jpeg_for_non_alpha_images() -> None:
    prepared = prepare_image_tensor(make_tensor_image(20, 10, 3))

    assert prepared.mime_type == "image/jpeg"
    assert prepared.data_uri.startswith("data:image/jpeg;base64,")


def test_prepare_image_tensor_resizes_large_images() -> None:
    prepared = prepare_image_tensor(make_tensor_image(1100, 1000, 3))

    assert prepared.width * prepared.height <= MAX_PIXELS
    assert prepared.width < 1100


def test_load_prompt_preserves_whitespace(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("line 1\n  line 2\n", encoding="utf-8")

    assert load_prompt(prompt_file) == "line 1\n  line 2\n"


def test_load_prompt_reads_caption_from_json(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompt.json"
    prompt_file.write_text('{"caption": "label the image carefully"}', encoding="utf-8")

    assert load_prompt(prompt_file, "json") == "label the image carefully"


def test_load_prompt_auto_detects_json_by_extension(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompt.json"
    prompt_file.write_text('{"caption": "auto detected"}', encoding="utf-8")

    assert load_prompt(prompt_file) == "auto detected"


def test_load_prompt_rejects_json_without_caption(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompt.json"
    prompt_file.write_text('{"wrong": "value"}', encoding="utf-8")

    with pytest.raises(BatchLabelError):
        load_prompt(prompt_file, "json")


def test_load_image_caption_reads_matching_json_sidecar(tmp_path: Path) -> None:
    image_path = tmp_path / "photo.png"
    create_image(image_path, (10, 10))
    caption_file = tmp_path / "photo.png.json"
    caption_file.write_text('{"caption": "describe the object"}', encoding="utf-8")

    assert load_image_caption(image_path) == "describe the object"


def test_load_image_caption_returns_none_when_missing(tmp_path: Path) -> None:
    image_path = tmp_path / "photo.png"
    create_image(image_path, (10, 10))

    assert load_image_caption(image_path) is None


def test_load_prompt_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(BatchLabelError):
        load_prompt(tmp_path / "missing.txt")


def test_build_chat_payload_has_one_system_and_one_user_message() -> None:
    prepared_image = PreparedImage(
        mime_type="image/jpeg",
        data_uri="data:image/jpeg;base64,AAAA",
        width=100,
        height=100,
    )

    payload = build_chat_payload(
        "gemma-4-test",
        "system prompt",
        USER_IMAGE_INSTRUCTION,
        prepared_image,
    )

    assert payload["model"] == "gemma-4-test"
    assert len(payload["messages"]) == 2
    assert payload["messages"][0] == {"role": "system", "content": "system prompt"}
    assert payload["messages"][1]["role"] == "user"
    assert payload["messages"][1]["content"][0]["text"] == USER_IMAGE_INSTRUCTION


def test_append_prompt_text_only_appends_non_empty_extra_text() -> None:
    assert append_prompt_text("base", "") == "base"
    assert append_prompt_text("base\n", "extra") == "base\n\nextra"


def test_format_batch_text_output_formats_single_and_multiple_results() -> None:
    assert format_batch_text_output(["one"]) == "one"
    assert format_batch_text_output(["one", "two"]) == "Image 1:\none\n\nImage 2:\ntwo"


def test_list_prompt_presets_returns_stable_order() -> None:
    assert list_prompt_presets() == [
        "short_caption",
        "detailed_description",
        "ocr_text_and_scene",
        "diffusion_json_blueprint",
        "nano_banana_json_blueprint",
    ]


@pytest.mark.parametrize("preset_id", list_prompt_presets())
def test_get_prompt_preset_text_returns_non_empty_text(preset_id: str) -> None:
    assert get_prompt_preset_text(preset_id)
