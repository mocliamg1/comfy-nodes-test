from __future__ import annotations

from typing import Any

import httpx

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
        http_client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(
            timeout=httpx.Timeout(timeout=timeout, connect=5.0)
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

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
        try:
            response = self._client.request(method, f"{self.base_url}{path}", json=json)
        except httpx.HTTPError as exc:
            raise LlamaServerError(f"Request to {path} failed: {exc}") from exc

        if response.status_code >= 400:
            body = response.text.strip()
            detail = f" {body}" if body else ""
            raise LlamaServerError(
                f"Server returned HTTP {response.status_code} for {path}.{detail}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise LlamaServerError(f"Server returned invalid JSON for {path}.") from exc

        if include_status:
            return payload, response.status_code
        return payload

    def maybe_unload_model(self, model: str) -> bool:
        try:
            response = self._client.request(
                "POST",
                f"{self.base_url}/models/unload",
                json={"model": model},
            )
        except httpx.HTTPError:
            return False

        return response.status_code < 400


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
