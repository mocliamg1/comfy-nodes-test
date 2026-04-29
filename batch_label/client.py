from __future__ import annotations

import json as json_module
from typing import Any
from urllib import error, request

from batch_label.types import LabelResponse, PreparedImage, ServerModelInfo


class LlamaServerError(Exception):
    """Raised for API and protocol errors."""


def build_chat_payload(
    model: str,
    system_prompt: str,
    user_text: str,
    prepared_image: PreparedImage,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_text,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": prepared_image.data_uri,
                        },
                    },
                ],
            },
        ],
    }


def extract_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)

    return ""


class LlamaServerClient:
    def __init__(
        self,
        base_url: str,
        timeout: float,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def close(self) -> None:
        return None

    def __enter__(self) -> "LlamaServerClient":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def get_models(self) -> list[ServerModelInfo]:
        payload = self._request_json("GET", "/models")
        data = payload.get("data")
        if not isinstance(data, list):
            raise LlamaServerError("Malformed /models response: missing data list.")

        models: list[ServerModelInfo] = []
        for item in data:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                models.append(ServerModelInfo(id=item["id"]))
        return models

    def label_image(
        self,
        model: str,
        system_prompt: str,
        user_text: str,
        prepared_image: PreparedImage,
    ) -> LabelResponse:
        payload = build_chat_payload(model, system_prompt, user_text, prepared_image)
        response_payload, status_code = self._request_json(
            "POST",
            "/chat/completions",
            json=payload,
            include_status=True,
        )

        try:
            message_content = response_payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LlamaServerError("Malformed chat response: missing assistant message content.") from exc

        text = extract_message_text(message_content).strip()
        if not text:
            raise LlamaServerError("Chat response did not contain label text.")

        return LabelResponse(text=text, raw_status=status_code, model_id=model)

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        include_status: bool = False,
    ) -> Any:
        body: bytes | None = None
        headers: dict[str, str] = {}
        if json is not None:
            body = json_module.dumps(json).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request_object = request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )

        try:
            with request.urlopen(request_object, timeout=self.timeout) as response:
                status_code = response.status
                response_body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace").strip()
            detail = f" {response_body}" if response_body else ""
            raise LlamaServerError(f"Server returned HTTP {exc.code} for {path}.{detail}") from exc
        except OSError as exc:
            raise LlamaServerError(f"Request to {path} failed: {exc}") from exc

        try:
            payload = json_module.loads(response_body)
        except ValueError as exc:
            raise LlamaServerError(f"Server returned invalid JSON for {path}.") from exc

        if include_status:
            return payload, status_code
        return payload

    def maybe_unload_model(self, model: str) -> bool:
        try:
            self._request_json(
                "POST",
                "/models/unload",
                json={"model": model},
            )
        except LlamaServerError:
            return False

        return True


def resolve_model_id(client: LlamaServerClient, requested_model: str | None) -> str:
    if requested_model:
        return requested_model

    models = client.get_models()
    if not models:
        raise LlamaServerError("The server did not return any models from /v1/models.")
    if len(models) > 1:
        available_models = ", ".join(model.id for model in models)
        raise LlamaServerError(
            "The server returned multiple models; pass --model explicitly. "
            f"Available models: {available_models}"
        )
    return models[0].id
