"""Thresholds from the PRD. They are starting values, tuned on golden clips."""

FLOW_LIMIT = 6.0
MIN_WINDOW_S = 3.0
MAX_WINDOW_SCORE_S = 10.0
OCCLUSION_DROP = 0.80
LIGHTING_VARIANCE_LIMIT = 40.0
PLACEABLE_SCORE = 70.0
WARNING_SCORE = 40.0
MAX_UPLOAD_S = 180.0
MAX_LONG_EDGE = 1920
MAX_SHORT_EDGE = 1080
MAX_RENDER_S = 15.0
TRACK_FPS = 12.0
JITTER_BUDGET = 2.0
MIN_SURFACE_FRACTION = 0.005
MAX_BAD_FRAME_FRACTION = 0.25
ALLOWED_CODECS = frozenset({"h264", "hevc", "h265", "mpeg4"})

WEIGHTS = {
    "surface_time": 0.30,
    "camera_motion": 0.25,
    "occlusion": 0.15,
    "lighting": 0.15,
    "shot_stability": 0.15,
}

SAM_MODEL = "sam-3.1"
SPARK_MODEL = "muse-spark-1.3"
IMAGE_MODEL = "muse-image-1.0"
META_BASE_URL = "https://api.meta.ai/v1"
OCCLUDER_PROMPT = "person"
