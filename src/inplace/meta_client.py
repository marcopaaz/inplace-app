from __future__ import annotations

import base64
import os
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TypeVar

import cv2
import numpy as np
from openai import APIConnectionError, APIStatusError, OpenAI, Stream
from openai.types.responses import Response, ResponseStreamEvent

from inplace.constants import IMAGE_MODEL, META_BASE_URL, SAM_MODEL
from inplace.errors import ModelError, PipelineError

T = TypeVar("T")


def should_retry(status_code: int) -> bool:
    return status_code == 429 or status_code >= 500


def load_project_env(path: Path | None = None) -> None:
    """Fill missing environment variables from the project .env file."""
    env_file = path if path is not None else Path.cwd() / ".env"
    if not env_file.is_file():
        return
    for raw in env_file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if name:
            os.environ.setdefault(name, value)


class MetaClient:
    """Calls the Meta Model API without letting the OpenAI SDK drop Meta-only fields."""

    def __init__(self, api_key: str | None = None) -> None:
        load_project_env()
        key = api_key if api_key is not None else os.environ.get("MODEL_API_KEY")
        if not key:
            raise PipelineError("MODEL_API_KEY is not set")
        self._client = OpenAI(base_url=META_BASE_URL, api_key=key)

    def upload(self, path: Path) -> str:
        def send() -> str:
            with path.open("rb") as handle:
                created = self._client.files.create(file=handle, purpose="user_data")
            return created.id

        return self._call(send)

    def segment_video(self, file_id: str, prompt: str) -> str:
        return self._read_stream(
            {
                "model": SAM_MODEL,
                "stream": True,
                "metadata": {"mask_encoding": "one_bit"},
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_video", "file_id": file_id},
                        ],
                    }
                ],
            }
        )

    def segment_image(self, image_bgr: np.ndarray, prompt: str) -> str:
        ok, encoded = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            raise PipelineError("Could not encode a spot-check frame")
        data_url = "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")
        return self._read_stream(
            {
                "model": SAM_MODEL,
                "stream": True,
                "metadata": {"mask_encoding": "one_bit"},
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": data_url},
                        ],
                    }
                ],
            }
        )

    def _read_stream(self, payload: dict[str, object]) -> str:
        delay = 1.0
        last: ModelError | None = None
        for attempt in range(3):
            try:
                return collect_output_text(self._post(payload, stream=True))
            except ModelError as exc:
                last = exc
                if not exc.retryable or attempt == 2:
                    if exc.retryable:
                        raise ModelError(f"{exc} (tried 3 times)", retryable=False) from exc
                    raise
            time.sleep(delay)
            delay *= 2
        if last is None:
            raise PipelineError("Meta API call failed")
        raise last

    def responses_text(
        self,
        *,
        model: str,
        content: list[dict[str, str]],
        schema: dict[str, object] | None = None,
        schema_name: str | None = None,
    ) -> str:
        payload: dict[str, object] = {
            "model": model,
            "input": [{"type": "message", "role": "user", "content": content}],
        }
        if schema is not None:
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": schema_name or "response",
                    "strict": True,
                    "schema": schema,
                }
            }
        return response_text(self._post(payload, stream=False))

    def generate_image(self, *, prompt: str, size: str) -> bytes:
        response = self._post(
            {
                "model": IMAGE_MODEL,
                "input": prompt,
                "tools": [
                    {
                        "type": "image_generation",
                        "enable_web_search": False,
                        "enable_image_search": False,
                        "enable_shell": False,
                        "reasoning_strength": "low",
                        "size": size,
                        "output_format": "png",
                    }
                ],
            },
            stream=False,
        )
        return image_bytes(response)

    def _post(self, payload: dict[str, object], *, stream: bool) -> Response | Stream[ResponseStreamEvent]:
        def send() -> Response | Stream[ResponseStreamEvent]:
            if stream:
                return self._client.post(
                    "/responses",
                    body=payload,
                    cast_to=ResponseStreamEvent,
                    stream=True,
                    stream_cls=Stream[ResponseStreamEvent],
                )
            return self._client.post("/responses", body=payload, cast_to=Response, stream=False)

        return self._call(send)

    def _call(self, fn: Callable[[], T], attempts: int = 3) -> T:
        delay = 1.0
        for attempt in range(attempts):
            try:
                return fn()
            except APIStatusError as exc:
                if not should_retry(exc.status_code) or attempt == attempts - 1:
                    raise
            except APIConnectionError:
                if attempt == attempts - 1:
                    raise
            time.sleep(delay)
            delay *= 2
        raise PipelineError("Meta API call failed")


_RETRYABLE_CODES = {"rate_limit_exceeded", "server_error", "vector_store_timeout"}


def collect_output_text(stream: Iterator[object]) -> str:
    deltas: list[str] = []
    done: list[str] = []
    completed = ""
    failure: tuple[str, bool] | None = None
    saw_terminal = False
    for event in stream:
        event_type = _field(event, "type")
        if event_type == "response.output_text.delta":
            delta = _field(event, "delta")
            if isinstance(delta, str):
                deltas.append(delta)
        elif event_type in {"response.output_text.done", "response.content_part.done"}:
            text = _field(event, "text")
            if not isinstance(text, str):
                part = _field(event, "part")
                text = _field(part, "text") if part is not None else None
            if isinstance(text, str) and text:
                done.append(text)
        elif event_type == "response.completed":
            saw_terminal = True
            completed = _text_from_response(_field(event, "response"))
        elif event_type in {"response.failed", "response.incomplete"}:
            saw_terminal = True
            failure = _failure_from_response(_field(event, "response"), event_type)
        elif event_type == "error":
            saw_terminal = True
            failure = _failure_from_event(event)
    if failure is not None:
        message, retryable = failure
        raise ModelError(message, retryable=retryable)
    if deltas:
        return "".join(deltas)
    if done:
        return done[-1]
    if completed:
        return completed
    if saw_terminal:
        return ""
    raise ModelError("The model returned no text", retryable=True)


def response_text(response: object) -> str:
    text = _text_from_response(response)
    if text:
        return text
    raise PipelineError("The model returned no text")


def _text_from_response(response: object) -> str:
    direct = _field(response, "output_text")
    if isinstance(direct, str) and direct:
        return direct
    chunks: list[str] = []
    output = _field(response, "output")
    if isinstance(output, list):
        for item in output:
            if _field(item, "type") != "message":
                continue
            content = _field(item, "content")
            if not isinstance(content, list):
                continue
            for block in content:
                if _field(block, "type") == "output_text":
                    text = _field(block, "text")
                    if isinstance(text, str):
                        chunks.append(text)
    return "".join(chunks)


def _failure_from_response(response: object, event_type: str) -> tuple[str, bool]:
    error = _field(response, "error")
    message = _field(error, "message")
    code = _field(error, "code")
    if isinstance(message, str) and message:
        return message, _retryable_code(code)
    if event_type == "response.incomplete":
        reason = _field(_field(response, "incomplete_details"), "reason")
        if isinstance(reason, str) and reason:
            return f"The model stopped early: {reason}", True
        return "The model stopped early", True
    return "The model failed", _retryable_code(code)


def _failure_from_event(event: object) -> tuple[str, bool]:
    message = _field(event, "message")
    code = _field(event, "code")
    text = message if isinstance(message, str) and message else "The model returned an error"
    return text, _retryable_code(code)


def _retryable_code(code: object) -> bool:
    return isinstance(code, str) and code in _RETRYABLE_CODES


def image_bytes(response: object) -> bytes:
    output = _field(response, "output")
    if isinstance(output, list):
        for item in output:
            if _field(item, "type") != "image_generation_call":
                continue
            result = _field(item, "result")
            if isinstance(result, str) and result:
                return base64.b64decode(result)
    raise PipelineError("Muse Image returned no image")


def _field(value: object, key: str) -> object | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)
