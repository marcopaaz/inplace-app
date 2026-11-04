from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from inplace.errors import StageFailed
from inplace.jobs import JobStore, cache_key

DOWNSTREAM: dict[str, tuple[str, ...]] = {
    "probe": ("signals", "plan", "spot", "score", "track", "creative", "warp", "composite", "report"),
    "signals": ("spot", "score", "track", "warp", "composite", "report"),
    "plan": ("spot", "score", "track", "creative", "warp", "composite", "report"),
    "spot": ("score", "track", "creative", "warp", "composite", "report"),
    "score": ("track", "creative", "warp", "composite", "report"),
    "track": ("warp", "composite", "report"),
    "creative": ("warp", "composite", "report"),
    "warp": ("composite", "report"),
    "composite": ("report",),
    "report": (),
}

PAID_DIRS: dict[str, tuple[str, ...]] = {
    "track": ("paid/sam",),
    "creative": ("paid/creative",),
    "plan": ("paid/spark",),
    "spot": ("paid/spot",),
}


def invalidate(job: JobStore, stage: str) -> None:
    for name in (stage, *DOWNSTREAM[stage]):
        job.clear_derived(name)
        for relative in PAID_DIRS.get(name, ()):
            job.remove(job.path(relative))


def run_cached(
    job: JobStore,
    *,
    stage: str,
    version: str,
    inputs: dict[str, str],
    outputs: list[str],
    force: bool,
    fn: Callable[[], None],
) -> Literal["ran", "cached"]:
    if force:
        invalidate(job, stage)
    key = cache_key(stage=stage, version=version, inputs=inputs)
    if not force and job.is_cached(stage, key, outputs):
        return "cached"
    job.clear_derived(stage)
    try:
        fn()
    except Exception as exc:
        for relative in outputs:
            job.remove(job.path(relative))
        job.write_error(stage, str(exc))
        raise
    missing = [relative for relative in outputs if not job.path(relative).is_file()]
    if missing:
        message = "missing outputs: " + ", ".join(missing)
        for relative in outputs:
            job.remove(job.path(relative))
        job.write_error(stage, message)
        raise StageFailed(message)
    job.write_manifest(stage=stage, cache_key=key, version=version, outputs=list(outputs))
    return "ran"
