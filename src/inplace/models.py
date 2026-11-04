from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

PlaceabilityStatus = Literal["placeable", "warning", "stopped"]
PlacementStatus = Literal["draft", "rendered", "failed"]
SignalName = Literal["surface_time", "camera_motion", "occlusion", "lighting", "shot_stability"]


class Video(BaseModel):
    id: str
    file_path: str
    duration_s: float
    fps: float
    width: int
    height: int
    codec: str
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SignalScores(BaseModel):
    surface_time: float
    camera_motion: float
    occlusion: float
    lighting: float
    shot_stability: float


class SignalSample(BaseModel):
    time_s: float
    flow: float
    luminance: float


class SignalScan(BaseModel):
    fps: float
    samples: list[SignalSample]
    cut_times: list[float]


class SceneWindow(BaseModel):
    start_s: float
    end_s: float
    surface_label: str
    sam_prompt: str


class MomentToAvoid(BaseModel):
    start_s: float
    end_s: float
    reason: str


class ScenePlan(BaseModel):
    windows: list[SceneWindow]
    moments_to_avoid: list[MomentToAvoid]


class SpotMeasurement(BaseModel):
    start_s: float
    end_s: float
    sam_prompt: str
    occlusion: float | None
    aspect_w: int | None
    aspect_h: int | None
    surface_found: bool


class SpotReport(BaseModel):
    measurements: list[SpotMeasurement]


class PlacementWindow(BaseModel):
    start_s: float
    end_s: float
    surface_label: str
    sam_prompt: str
    thumbnail: str | None = None
    window_score: float
    signal_scores: SignalScores
    aspect_w: int
    aspect_h: int


class RejectedWindow(BaseModel):
    start_s: float
    end_s: float
    surface_label: str
    reason: str


class PlaceabilityReport(BaseModel):
    video_id: str
    score: float
    status: PlaceabilityStatus
    reasons: list[str]
    windows: list[PlacementWindow]
    rejected: list[RejectedWindow] = Field(default_factory=list)


class BrandBrief(BaseModel):
    name: str
    description: str
    style: str


class Placement(BaseModel):
    video_id: str
    window_index: int
    start_s: float
    end_s: float
    surface_label: str
    sam_prompt: str
    aspect_w: int
    aspect_h: int
    brand: BrandBrief
    creative_index: int
    chosen_creative_path: str
    status: PlacementStatus


class RenderRecord(BaseModel):
    placement_id: str
    output_path: str
    jitter_rms: float
    qa_flags: list[str] = Field(default_factory=list)
    rerender_count: int = 0


class Visibility(BaseModel):
    render_id: str
    seconds_on_screen: float
    average_share_of_frame: float
    share_of_time_occluded: float


class StageManifest(BaseModel):
    stage: str
    version: str
    cache_key: str
    status: Literal["ok"] = "ok"
    outputs: list[str]
