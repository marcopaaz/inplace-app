import pytest

from inplace.errors import ModelError
from inplace.meta_client import MetaClient, collect_output_text


class Event:
    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)


def test_deltas_are_joined() -> None:
    text = collect_output_text(
        [
            Event(type="response.output_text.delta", delta="hel"),
            Event(type="response.output_text.delta", delta="lo"),
        ]
    )
    assert text == "hello"


def test_completed_event_supplies_text_when_there_are_no_deltas() -> None:
    text = collect_output_text(
        [
            Event(
                type="response.completed",
                response=Event(
                    output=[
                        Event(
                            type="message",
                            content=[Event(type="output_text", text="<mask>")],
                        )
                    ]
                ),
            )
        ]
    )
    assert text == "<mask>"


def test_completed_event_with_no_text_is_an_empty_mask() -> None:
    assert collect_output_text([Event(type="response.completed", response=Event(output=[]))]) == ""


def test_failed_event_keeps_the_model_message() -> None:
    with pytest.raises(ModelError, match="Slow down") as caught:
        collect_output_text(
            [
                Event(
                    type="response.failed",
                    response=Event(status="failed", error=Event(code="rate_limit_exceeded", message="Slow down")),
                )
            ]
        )
    assert caught.value.retryable


def test_image_error_is_not_retried() -> None:
    with pytest.raises(ModelError, match="bad image") as caught:
        collect_output_text([Event(type="error", code="invalid_image", message="bad image")])
    assert not caught.value.retryable


def test_empty_stream_can_be_retried() -> None:
    with pytest.raises(ModelError, match="no text") as caught:
        collect_output_text([])
    assert caught.value.retryable


def test_rate_limit_in_the_stream_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("inplace.meta_client.time.sleep", lambda _seconds: None)
    client = MetaClient(api_key="test")
    calls = {"n": 0}

    def post(payload: dict[str, object], *, stream: bool) -> list[Event]:
        del payload, stream
        calls["n"] += 1
        if calls["n"] == 1:
            return [
                Event(
                    type="response.failed",
                    response=Event(error=Event(code="rate_limit_exceeded", message="Slow down")),
                )
            ]
        return [Event(type="response.output_text.delta", delta="<mask>")]

    monkeypatch.setattr(client, "_post", post)
    assert client.segment_video("file", "door") == "<mask>"
    assert calls["n"] == 2
