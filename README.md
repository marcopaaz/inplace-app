# Inplace

Inplace places a brand creative into a video that was already filmed. It scores whether a clip can carry a placement, finds a flat surface, generates posters, and composites one so it follows the camera and stays behind people who walk in front of it.

The product spec is in [PRD.md](PRD.md).

## What it does

1. **Check.** Probe the file, measure camera motion and cuts, ask Muse Spark for a scene plan, spot-check surfaces with SAM 3.1, and write a placeability score (0–100). Videos below the bar stop there, with a reason.
2. **Render.** Track the chosen surface and people through the window, generate three posters with Muse Image, warp the chosen poster onto the surface, composite it with occlusion, and write a visibility report.

Each stage writes its output under `jobs/<id>/`. Later runs reuse those files unless you pass `--force`.

## Requirements

- Python 3.12+
- `ffmpeg` and `ffprobe` on `PATH`
- A [Meta Model API](https://dev.meta.ai) key with access to Muse Spark 1.3, SAM 3.1, and Muse Image 1.0

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Put the key in a local `.env` file. That file is gitignored and is not part of this repository.

```bash
MODEL_API_KEY=your-key
```

An existing environment variable wins over the file.

## Usage

Score a video:

```bash
inplace check path/to/clip.mp4
```

The command prints a job path, a score, and numbered windows. Render one window with a sample brand (`northline`, `lumen soda`, or `paper orbit`) or your own brief:

```bash
inplace render <job-id> --window 0 --brand northline
inplace render <job-id> --window 0 --brand "Acme" --description "A fictional coffee roaster." --style "flat poster, warm brown, simple wordmark"
```

`--creative` picks poster 0, 1, or 2 (default 0). The finished video is `jobs/<id>/render/output.mp4`.

Open the local page (upload, score, pick a window and poster, render):

```bash
inplace ui
```

Then open http://127.0.0.1:8000.

Clips must be MP4 (H.264, HEVC, or MPEG-4), at most 3 minutes, and at most 1920×1080. Renders cover at most 15 seconds of a placeable window.

## Tests

```bash
pytest
```

Paid API calls are not part of the test suite. Job folders, sample footage, `.env`, and `.cursor` stay on your machine.
