# Inplace PRD

Oct 27, 2026 · @Marco Paz

## Overview

Inplace lets a creator turn videos they already published into sponsored placements: it checks which videos can carry an ad, finds where, and renders a brand creative into the scene so it looks like it was always there.

**Problem.** Product placement is decided at shooting time and needs agencies, so a creator's back catalog earns nothing new and small brands are priced out entirely.

**Goals**

- Tell a creator, in under a minute per video, whether a video is placeable and where.
- Render a placement that survives camera movement and people passing in front of it.
- Give brands a number they can buy on: seconds on screen and share of frame.

**Non-goals**

- Audience or contextual targeting, bidding, ad serving or payments.
- 3D product insertion (a can on a table); v1 places flat creatives on flat surfaces only.
- Live video or long-form content over 3 minutes.

## Users

The creator is the v1 user; the brand only shows up as a brief and as the reader of the visibility report.

| User | Wants | Inplace gives them |
| --- | --- | --- |
| Creator (primary) | New revenue from old videos without re-shooting or hurting the viewer experience | Placeability score per video, placement preview, final render with a Sponsored tag |
| Brand (secondary) | Cheap, native-looking exposure and proof of what they got | A creative generated from their brief and a visibility report |
| Platform or network (later) | Ad inventory that doesn't interrupt viewing | A channel-wide inventory report of placeable videos and sellable seconds |

## Core flow

One video goes from upload to sponsored render in seven steps; the creator decides twice (keep going after the score, approve the placement).

1. **Upload.** Creator uploads a clip (MP4, up to 3 minutes).
2. **Placeability check.** Inplace returns a 0–100 score, the reasons, and placeable windows. Below the threshold, the flow stops with the reason shown.
3. **Choose a window.** Creator picks one window (time range + surface) from the list, shown on a timeline with a thumbnail.
4. **Brand brief.** Creator enters a brand name, one line of description and a style hint, or picks a sample brand.
5. **Creative.** Muse Image generates 3 creatives sized to the surface; creator picks one.
6. **Render.** Inplace tracks, warps and composites; the creator watches a before/after preview and approves or re-renders.
7. **Deliver.** Download the rendered MP4 and the visibility report.

Channel scan (P1) runs step 2 over many videos at once and ends in the inventory report.

## Functional requirements

Thirteen requirements are P0 and make up the hackathon demo; the rest only get built if Phase 2 lands early.

| ID | Requirement | Priority |
| --- | --- | --- |
| FR-1 | Accept MP4 uploads up to 3 minutes and 1080p; reject anything else with a clear message | P0 must |
| FR-2 | Run the placeability check and return score, reasons and windows | P0 must |
| FR-3 | Stop the flow below the score threshold and show the main reason in plain language | P0 must |
| FR-4 | Each window carries start, end, surface label, SAM prompt and a thumbnail | P0 must |
| FR-5 | Take a brand brief (name, one-line description, style hint) or a sample fictional brand | P0 must |
| FR-6 | Generate 3 creatives per brief at the surface's aspect ratio with Muse Image | P0 must |
| FR-7 | Track the chosen surface and its occluders through the window with SAM 3.1 | P0 must |
| FR-8 | Warp the creative onto the surface every frame using a smoothed homography | P0 must |
| FR-9 | Draw occluders (people, hands, objects) in front of the creative | P0 must |
| FR-10 | Blend the creative with the surface's shading and feather its edges | P0 must |
| FR-11 | Stamp a Sponsored tag on every render; it can't be turned off | P0 must |
| FR-12 | Show a before/after preview with a slider before download | P0 must |
| FR-13 | Produce a visibility report: seconds on screen, average share of frame, share of time occluded | P0 must |
| FR-14 | Muse Spark reviews rendered frames and triggers one automatic re-render when it flags drift | P1 should |
| FR-15 | Channel scan: score many videos at once and output the inventory report | P1 should |
| FR-16 | Place the same brand on the video thumbnail with Muse Image editing | P2 later |
| FR-17 | Several brands compete for one video's windows (marketplace view) | P2 later |

## Placeability score

The score is computed per window, and the video's score is its best window; all weights and thresholds below are starting values to tune on the golden clips.

```latex
\text{score} = 100 \cdot \sum_i w_i \, s_i \qquad s_i \in [0, 1]
```

| Signal | Normalized score (1 = best) | Weight | Hard gate |
| --- | --- | --- | --- |
| Surface time on screen | Window length, capped at 10 s | 0.30 | Under 3 s: window dropped |
| Camera motion | Inverse of median optical flow magnitude | 0.25 | Shake above the limit: window dropped |
| Occlusion | Best near 10–30% of frames occluded, lower at 0% or above 60% | 0.15 | Above 80%: window dropped |
| Lighting stability | Inverse of luminance variance on the surface | 0.15 | None |
| Shot stability | 1 if the window has no cut, else split it at the cut | 0.15 | None |
| Moments to avoid | Not scored: Muse Spark removes those time ranges before scoring | n/a | Flagged range: excluded |

**Thresholds.** 70 or more: placeable. 40 to 69: placeable with a warning shown to the creator. Under 40: flow stops (FR-3).

**Reasons.** The two lowest-scoring signals become the reasons shown to the creator, phrased by Muse Spark in plain language.

## Non-functional requirements

These are targets for a 1-minute, 1080p, 30 fps clip; measure them on the golden clips and adjust.

| Area | Target |
| --- | --- |
| Placeability check | Result in under 60 s |
| Render | Under 5 minutes for a 10 s window |
| Model cost | Under $1 per rendered minute; known parts are SAM 3.1 at $0.20 per 1k frames (about $0.36 for a full minute) and Muse Image at $0.01 per creative |
| Visual quality | No visible jitter at normal playback; occluders never appear behind the creative |
| Re-runs | Each stage caches its output so a re-render never re-calls SAM or Muse Image |
| Privacy | Uploads stay in the job folder and are deleted after 7 days; no face or person identification, people are only masked as occluders |
| Disclosure | Every render carries the Sponsored tag (FR-11) |

## Data model

Five records cover the whole flow; each one maps to a JSON file in the job folder for the hackathon, and to a table later.

| Record | Key fields | Written by |
| --- | --- | --- |
| Video | id, file path, duration, fps, resolution, uploaded at | Upload |
| Placeability report | video id, score, reasons, signal scores, windows (start, end, surface label, SAM prompt, thumbnail, window score) | Placeability check |
| Placement | video id, window, brand brief, chosen creative path, status | Creator choices |
| Render | placement id, output path, QA flags, re-render count | Render and QA |
| Visibility metrics | render id, seconds on screen, average share of frame, share of time occluded | Report |

The inventory report (FR-15) is a rollup of placeability reports per channel: placeable videos, total placeable seconds, average score.

## Success metrics

For the hackathon, success is a demo that holds up live; for the product, it's creators approving placements without re-renders.

**Hackathon**

- 3 golden clips render with no visible jitter, including one where a person walks in front of the placement.
- The placeability check correctly rejects at least one bad clip (shaky or no surface) in the demo.
- The full flow runs live in under 5 minutes on a fresh clip.

**Product (after the hackathon)**

- First-render approval rate: share of placements the creator approves without re-rendering.
- Score accuracy: share of videos scored placeable that end in an approved render.
- Placeable seconds per scanned channel, the inventory a creator could sell.

## Scope and open questions

The MVP is FR-1 to FR-13 on flat surfaces in short clips; the build order and demo script live in Inplace: Virtual Product Placement.

**Later:** QA loop and channel scan (P1), thumbnail placement and multi-brand marketplace (P2), 3D object insertion, longer videos, a brand-side portal.

**Open questions**

- [ ] Is 3 minutes the right upload cap, or should the demo stick to clips under 1 minute?
- [ ] What request and mask format does SAM 3.1 return for video on the Meta Model API?
- [ ] Does Muse Spark's video input cover a full 3-minute clip in one call, or does it need sampled frames?
- [ ] Are the score weights and thresholds right? Tune on the golden clips.
- [ ] Does the GumGum employment agreement allow this as a public side project?
