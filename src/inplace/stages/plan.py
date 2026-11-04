from __future__ import annotations

import json

from openai import APIStatusError

from inplace.constants import SPARK_MODEL
from inplace.errors import PipelineError
from inplace.jobs import JobStore, hash_file
from inplace.meta_client import MetaClient
from inplace.models import ScenePlan
from inplace.runner import run_cached

PLAN_VERSION = "2"

SCENE_PROMPT = (
    "You watch a video and choose where a flat poster, screen, or wall decal could sit. "
    "Return JSON with windows and moments_to_avoid. "
    "Return at most 3 windows. Each window is between 3 and 15 seconds. "
    "Do not mark the whole video as one window. Pick the steadiest stretch of each surface. "
    "Each window has start_s, end_s, surface_label, and sam_prompt. "
    "sam_prompt is a short noun phrase naming one flat surface, such as \"blank wall behind the desk\". "
    "moments_to_avoid lists start_s, end_s, and reason for sensitive or off-topic stretches. "
    "Use an empty list when there are none. Do not invent a brand and do not describe a real logo."
)

SCENE_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "windows": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "start_s": {"type": "number"},
                    "end_s": {"type": "number"},
                    "surface_label": {"type": "string"},
                    "sam_prompt": {"type": "string"},
                },
                "required": ["start_s", "end_s", "surface_label", "sam_prompt"],
            },
        },
        "moments_to_avoid": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "start_s": {"type": "number"},
                    "end_s": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["start_s", "end_s", "reason"],
            },
        },
    },
    "required": ["windows", "moments_to_avoid"],
}


def parse_scene_plan(text: str) -> ScenePlan:
    try:
        return ScenePlan.model_validate(json.loads(json_object(text)))
    except (json.JSONDecodeError, ValueError) as exc:
        raise PipelineError(f"Spark did not return a scene plan: {exc}") from exc


def json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise PipelineError("Spark did not return JSON")
    return text[start : end + 1]


def run_plan(job: JobStore, client: MetaClient, force: set[str]) -> str:
    source = job.path("source.mp4")

    def produce() -> None:
        raw_path = job.path(f"paid/spark/scene_plan-{PLAN_VERSION}.txt")
        if raw_path.is_file():
            text = raw_path.read_text()
        else:
            file_id = client.upload(source)
            try:
                text = client.responses_text(
                    model=SPARK_MODEL,
                    content=[
                        {"type": "input_text", "text": SCENE_PROMPT},
                        {"type": "input_file", "file_id": file_id},
                    ],
                    schema=SCENE_SCHEMA,
                    schema_name="scene_plan",
                )
            except APIStatusError as exc:
                if exc.status_code != 400:
                    raise
                text = client.responses_text(
                    model=SPARK_MODEL,
                    content=[
                        {"type": "input_text", "text": SCENE_PROMPT},
                        {"type": "input_file", "file_id": file_id},
                    ],
                )
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(text)
        job.write_model("scene_plan.json", parse_scene_plan(text))

    return run_cached(
        job,
        stage="plan",
        version=PLAN_VERSION,
        inputs={"source": hash_file(source), "model": SPARK_MODEL},
        outputs=["scene_plan.json"],
        force="plan" in force,
        fn=produce,
    )
