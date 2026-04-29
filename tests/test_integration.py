import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

import batch_label.comfy_node as comfy_node_module
from batch_label.cli import main
from batch_label.comfy_node import BatchLabelLlamaVisionText
from batch_label.presets import get_prompt_preset_text


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


def make_tensor_batch(batch_size: int, width: int = 4, height: int = 3, channels: int = 3) -> FakeTensor:
    pixel = [0.1, 0.5, 0.9, 0.7][:channels]
    image = [[list(pixel) for _ in range(width)] for _ in range(height)]
    return FakeTensor([[row[:] for row in image] for _ in range(batch_size)])


def create_image(path: Path, size: tuple[int, int], mode: str = "RGB") -> None:
    color = (0, 128, 255, 255) if "A" in mode else (0, 128, 255)
    Image.new(mode, size, color=color).save(path)


class MockLlamaServer:
    def __init__(
        self,
        models: list[str],
        chat_responses: list[dict[str, Any]],
        *,
        unload_status: int = 200,
    ) -> None:
        self.models = models
        self.chat_responses = chat_responses
        self.requests: list[dict[str, Any]] = []
        self.unload_requests: list[dict[str, Any]] = []
        self.unload_status = unload_status
        self._lock = threading.Lock()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._build_handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}/v1"

    def start(self) -> "MockLlamaServer":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()

    def _build_handler(self) -> type[BaseHTTPRequestHandler]:
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path != "/v1/models":
                    self.send_error(404)
                    return

                self._send_json(200, {"data": [{"id": model_id} for model_id in parent.models]})

            def do_POST(self) -> None:  # noqa: N802
                content_length = int(self.headers["Content-Length"])
                body = self.rfile.read(content_length)
                payload = json.loads(body.decode("utf-8"))

                if self.path == "/v1/chat/completions":
                    with parent._lock:
                        parent.requests.append(payload)
                        response = parent.chat_responses.pop(0)
                    self._send_json(response["status"], response["body"])
                    return

                if self.path == "/v1/models/unload":
                    with parent._lock:
                        parent.unload_requests.append(payload)
                    self._send_json(parent.unload_status, {"ok": parent.unload_status < 400})
                    return

                self.send_error(404)

            def log_message(self, *_args: object) -> None:
                return

            def _send_json(self, status: int, payload: dict[str, Any]) -> None:
                encoded = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        return Handler


class MockComfyServer:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._build_handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def free_url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}/free"

    def start(self) -> "MockComfyServer":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()

    def _build_handler(self) -> type[BaseHTTPRequestHandler]:
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                if self.path != "/free":
                    self.send_error(404)
                    return

                content_length = int(self.headers["Content-Length"])
                body = self.rfile.read(content_length)
                payload = json.loads(body.decode("utf-8"))
                with parent._lock:
                    parent.requests.append(payload)
                self._send_json(200, {"ok": True})

            def log_message(self, *_args: object) -> None:
                return

            def _send_json(self, status: int, payload: dict[str, Any]) -> None:
                encoded = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        return Handler


def test_cli_labels_images_and_uses_fresh_chat_requests(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    create_image(images_dir / "one.jpg", (1800, 1200))
    create_image(images_dir / "two.png", (400, 400), mode="RGBA")
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("Describe the image in one short label.\n", encoding="utf-8")
    (images_dir / "two.png.json").write_text(
        '{"caption": "Focus on transparent elements."}',
        encoding="utf-8",
    )

    server = MockLlamaServer(
        models=["qwen-3.5-vl"],
        chat_responses=[
            {"status": 200, "body": {"choices": [{"message": {"content": "label one"}}]}},
            {"status": 200, "body": {"choices": [{"message": {"content": "label two"}}]}},
        ],
    ).start()

    stderr = io.StringIO()
    try:
        exit_code = main(
            [
                "--images",
                str(images_dir),
                "--prompt-file",
                str(prompt_file),
                "--base-url",
                server.base_url,
                "--verbose",
            ],
            stderr=stderr,
        )
    finally:
        server.stop()

    assert exit_code == 0
    assert (images_dir / "one.txt").read_text(encoding="utf-8") == "label one\n"
    assert (images_dir / "two.txt").read_text(encoding="utf-8") == "label two\n"
    assert len(server.requests) == 2
    assert server.requests[0]["messages"][0]["content"] == "Describe the image in one short label.\n"
    assert (
        server.requests[1]["messages"][0]["content"]
        == "Describe the image in one short label.\n\nFocus on transparent elements."
    )


def test_cli_skips_existing_sidecars_by_default(tmp_path: Path) -> None:
    image_path = tmp_path / "photo.jpg"
    create_image(image_path, (100, 100))
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("label", encoding="utf-8")
    sidecar_path = tmp_path / "photo.txt"
    sidecar_path.write_text("existing\n", encoding="utf-8")

    server = MockLlamaServer(models=["gemma-4-vision"], chat_responses=[]).start()
    stderr = io.StringIO()
    try:
        exit_code = main(
            [
                "--images",
                str(image_path),
                "--prompt-file",
                str(prompt_file),
                "--base-url",
                server.base_url,
            ],
            stderr=stderr,
        )
    finally:
        server.stop()

    assert exit_code == 0
    assert sidecar_path.read_text(encoding="utf-8") == "existing\n"
    assert server.requests == []


def test_cli_force_overwrites_existing_sidecars(tmp_path: Path) -> None:
    image_path = tmp_path / "photo.jpg"
    create_image(image_path, (100, 100))
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("label", encoding="utf-8")
    sidecar_path = tmp_path / "photo.txt"
    sidecar_path.write_text("existing\n", encoding="utf-8")

    server = MockLlamaServer(
        models=["gemma-4-vision"],
        chat_responses=[{"status": 200, "body": {"choices": [{"message": {"content": "new label"}}]}}],
    ).start()
    try:
        exit_code = main(
            [
                "--images",
                str(image_path),
                "--prompt-file",
                str(prompt_file),
                "--base-url",
                server.base_url,
                "--force",
            ],
        )
    finally:
        server.stop()

    assert exit_code == 0
    assert sidecar_path.read_text(encoding="utf-8") == "new label\n"
    assert len(server.requests) == 1


def test_cli_continues_after_one_image_fails(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    create_image(images_dir / "one.jpg", (100, 100))
    create_image(images_dir / "two.jpg", (100, 100))
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("label", encoding="utf-8")

    server = MockLlamaServer(
        models=["qwen-3.5-vl"],
        chat_responses=[
            {"status": 500, "body": {"error": {"message": "boom"}}},
            {"status": 200, "body": {"choices": [{"message": {"content": "second label"}}]}},
        ],
    ).start()

    stderr = io.StringIO()
    try:
        exit_code = main(
            [
                "--images",
                str(images_dir),
                "--prompt-file",
                str(prompt_file),
                "--base-url",
                server.base_url,
            ],
            stderr=stderr,
        )
    finally:
        server.stop()

    assert exit_code == 1
    assert not (images_dir / "one.txt").exists()
    assert (images_dir / "two.txt").read_text(encoding="utf-8") == "second label\n"
    assert "Failed" in stderr.getvalue()


def test_cli_errors_when_multiple_models_are_available(tmp_path: Path) -> None:
    image_path = tmp_path / "photo.jpg"
    create_image(image_path, (100, 100))
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("label", encoding="utf-8")

    server = MockLlamaServer(
        models=["qwen-3.5-vl", "gemma-4-vision"],
        chat_responses=[],
    ).start()

    stderr = io.StringIO()
    try:
        exit_code = main(
            [
                "--images",
                str(image_path),
                "--prompt-file",
                str(prompt_file),
                "--base-url",
                server.base_url,
            ],
            stderr=stderr,
        )
    finally:
        server.stop()

    assert exit_code == 1
    assert "multiple models" in stderr.getvalue()


def test_cli_accepts_single_image_and_passes_explicit_model_unchanged(tmp_path: Path) -> None:
    image_path = tmp_path / "photo.jpg"
    create_image(image_path, (100, 100))
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("label", encoding="utf-8")

    server = MockLlamaServer(
        models=["unused-model"],
        chat_responses=[{"status": 200, "body": {"choices": [{"message": {"content": "single label"}}]}}],
    ).start()

    try:
        exit_code = main(
            [
                "--images",
                str(image_path),
                "--prompt-file",
                str(prompt_file),
                "--base-url",
                server.base_url,
                "--model",
                "gemma-4-custom",
            ],
        )
    finally:
        server.stop()

    assert exit_code == 0
    assert (tmp_path / "photo.txt").read_text(encoding="utf-8") == "single label\n"
    assert server.requests[0]["model"] == "gemma-4-custom"


def test_cli_writes_recursive_captions_into_root_image_folder(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    nested_dir = images_dir / "nested"
    nested_dir.mkdir(parents=True)
    create_image(nested_dir / "photo.jpg", (100, 100))
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("label", encoding="utf-8")

    server = MockLlamaServer(
        models=["qwen-3.5-vl"],
        chat_responses=[{"status": 200, "body": {"choices": [{"message": {"content": "nested label"}}]}}],
    ).start()

    try:
        exit_code = main(
            [
                "--images",
                str(images_dir),
                "--prompt-file",
                str(prompt_file),
                "--base-url",
                server.base_url,
            ],
        )
    finally:
        server.stop()

    assert exit_code == 0
    assert (images_dir / "nested__photo.txt").read_text(encoding="utf-8") == "nested label\n"
    assert not (nested_dir / "photo.txt").exists()


def test_comfy_node_labels_single_image_and_cleans_up(monkeypatch: pytest.MonkeyPatch) -> None:
    llama_server = MockLlamaServer(
        models=["qwen-3.5-vl"],
        chat_responses=[{"status": 200, "body": {"choices": [{"message": {"content": "short label"}}]}}],
    ).start()
    comfy_server = MockComfyServer().start()
    monkeypatch.setattr(comfy_node_module, "COMFYUI_FREE_URL", comfy_server.free_url)

    try:
        node = BatchLabelLlamaVisionText()
        result = node.run(
            make_tensor_batch(1),
            "short_caption",
            "",
            base_url=llama_server.base_url,
            model="qwen-3.5-vl",
        )
    finally:
        comfy_server.stop()
        llama_server.stop()

    assert result == ("short label",)
    assert llama_server.unload_requests == [{"model": "qwen-3.5-vl"}]
    assert comfy_server.requests == [{"unload_models": True, "free_memory": False}]


def test_comfy_node_auto_resolves_single_model(monkeypatch: pytest.MonkeyPatch) -> None:
    llama_server = MockLlamaServer(
        models=["gemma-4-vision"],
        chat_responses=[{"status": 200, "body": {"choices": [{"message": {"content": "resolved"}}]}}],
    ).start()
    comfy_server = MockComfyServer().start()
    monkeypatch.setattr(comfy_node_module, "COMFYUI_FREE_URL", comfy_server.free_url)

    try:
        node = BatchLabelLlamaVisionText()
        result = node.run(make_tensor_batch(1), "short_caption", "", base_url=llama_server.base_url)
    finally:
        comfy_server.stop()
        llama_server.stop()

    assert result == ("resolved",)
    assert llama_server.requests[0]["model"] == "gemma-4-vision"


def test_comfy_node_errors_when_multiple_models_are_available(monkeypatch: pytest.MonkeyPatch) -> None:
    llama_server = MockLlamaServer(models=["one", "two"], chat_responses=[]).start()
    comfy_server = MockComfyServer().start()
    monkeypatch.setattr(comfy_node_module, "COMFYUI_FREE_URL", comfy_server.free_url)

    try:
        node = BatchLabelLlamaVisionText()
        with pytest.raises(Exception, match="multiple models"):
            node.run(make_tensor_batch(1), "short_caption", "", base_url=llama_server.base_url)
    finally:
        comfy_server.stop()
        llama_server.stop()


def test_comfy_node_combines_system_prompt_and_user_instructions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llama_server = MockLlamaServer(
        models=["qwen-3.5-vl"],
        chat_responses=[{"status": 200, "body": {"choices": [{"message": {"content": "ok"}}]}}],
    ).start()
    comfy_server = MockComfyServer().start()
    monkeypatch.setattr(comfy_node_module, "COMFYUI_FREE_URL", comfy_server.free_url)

    try:
        node = BatchLabelLlamaVisionText()
        node.run(
            make_tensor_batch(1),
            "short_caption",
            "Mention the background.",
            system_prompt_custom="Focus on color palette.",
            base_url=llama_server.base_url,
            model="qwen-3.5-vl",
        )
    finally:
        comfy_server.stop()
        llama_server.stop()

    payload = llama_server.requests[0]
    assert payload["messages"][0]["content"] == (
        f"{get_prompt_preset_text('short_caption')}\n\nFocus on color palette."
    )
    assert payload["messages"][1]["content"][0]["text"].endswith("\n\nMention the background.")


def test_comfy_node_processes_multi_image_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    llama_server = MockLlamaServer(
        models=["qwen-3.5-vl"],
        chat_responses=[
            {"status": 200, "body": {"choices": [{"message": {"content": "label one"}}]}},
            {"status": 200, "body": {"choices": [{"message": {"content": "label two"}}]}},
        ],
    ).start()
    comfy_server = MockComfyServer().start()
    monkeypatch.setattr(comfy_node_module, "COMFYUI_FREE_URL", comfy_server.free_url)

    try:
        node = BatchLabelLlamaVisionText()
        result = node.run(
            make_tensor_batch(2),
            "short_caption",
            "",
            base_url=llama_server.base_url,
            model="qwen-3.5-vl",
        )
    finally:
        comfy_server.stop()
        llama_server.stop()

    assert result == ("Image 1:\nlabel one\n\nImage 2:\nlabel two",)
    assert len(llama_server.requests) == 2


def test_comfy_node_raises_on_malformed_chat_response(monkeypatch: pytest.MonkeyPatch) -> None:
    llama_server = MockLlamaServer(
        models=["qwen-3.5-vl"],
        chat_responses=[{"status": 200, "body": {"wrong": "shape"}}],
    ).start()
    comfy_server = MockComfyServer().start()
    monkeypatch.setattr(comfy_node_module, "COMFYUI_FREE_URL", comfy_server.free_url)

    try:
        node = BatchLabelLlamaVisionText()
        with pytest.raises(Exception, match="Failed to label image 1"):
            node.run(make_tensor_batch(1), "short_caption", "", base_url=llama_server.base_url, model="qwen-3.5-vl")
    finally:
        comfy_server.stop()
        llama_server.stop()


def test_comfy_node_raises_on_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    comfy_server = MockComfyServer().start()
    monkeypatch.setattr(comfy_node_module, "COMFYUI_FREE_URL", comfy_server.free_url)

    try:
        node = BatchLabelLlamaVisionText()
        with pytest.raises(Exception):
            node.run(make_tensor_batch(1), "short_caption", "", base_url="http://127.0.0.1:1/v1", model="qwen-3.5-vl", timeout=1.0)
    finally:
        comfy_server.stop()
