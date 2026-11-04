class PipelineError(Exception):
    """A failure the creator can act on."""


class ModelError(PipelineError):
    """The model stream failed or ended without a usable response."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class VideoRejected(PipelineError):
    """The upload broke a hard rule."""


class StageFailed(PipelineError):
    """A stage finished without the files the next stage needs."""
