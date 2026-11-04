from __future__ import annotations

from dataclasses import dataclass
from typing import Never

from inplace.constants import (
    FLOW_LIMIT,
    LIGHTING_VARIANCE_LIMIT,
    MAX_RENDER_S,
    MAX_WINDOW_SCORE_S,
    MIN_WINDOW_S,
    OCCLUSION_DROP,
    PLACEABLE_SCORE,
    WARNING_SCORE,
    WEIGHTS,
)
from inplace.models import (
    PlaceabilityReport,
    PlaceabilityStatus,
    PlacementWindow,
    RejectedWindow,
    ScenePlan,
    SignalName,
    SignalScan,
    SignalScores,
    SpotReport,
)


@dataclass
class WindowInput:
    start_s: float
    end_s: float
    surface_label: str
    sam_prompt: str
    flow: float
    luminance_variance: float
    occlusion: float
    has_cut: bool
    aspect_w: int
    aspect_h: int
    surface_found: bool


@dataclass
class ScoredWindow:
    kept: bool
    drop_reason: str | None
    score: float
    signals: SignalScores
    start_s: float
    end_s: float
    surface_label: str
    sam_prompt: str
    aspect_w: int
    aspect_h: int
    occlusion_fraction: float


def subtract_intervals(
    start: float,
    end: float,
    blocks: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    pieces = [(start, end)]
    for block_start, block_end in sorted(blocks):
        nxt: list[tuple[float, float]] = []
        for piece_start, piece_end in pieces:
            if block_end <= piece_start or block_start >= piece_end:
                nxt.append((piece_start, piece_end))
                continue
            if block_start > piece_start:
                nxt.append((piece_start, min(block_start, piece_end)))
            if block_end < piece_end:
                nxt.append((max(block_end, piece_start), piece_end))
        pieces = [(piece_start, piece_end) for piece_start, piece_end in nxt if piece_end - piece_start > 1e-3]
    return pieces


def split_on_cuts(start: float, end: float, cuts: list[float]) -> list[tuple[float, float]]:
    points = [start, *[cut for cut in cuts if start < cut < end], end]
    return [(left, right) for left, right in zip(points, points[1:]) if right - left > 1e-3]


def score_window(window: WindowInput) -> ScoredWindow:
    duration = window.end_s - window.start_s
    signals = SignalScores(
        surface_time=_surface_signal(duration),
        camera_motion=_motion_signal(window.flow),
        occlusion=_occlusion_signal(window.occlusion),
        lighting=_lighting_signal(window.luminance_variance),
        shot_stability=0.0 if window.has_cut else 1.0,
    )
    drop_reason = _drop_reason(window, duration)
    if drop_reason is not None:
        return _scored(window, kept=False, score=0.0, signals=signals, drop_reason=drop_reason)
    weighted = sum(WEIGHTS[name] * getattr(signals, name) for name in WEIGHTS)
    return _scored(window, kept=True, score=float(round(100 * weighted)), signals=signals, drop_reason=None)


def summarize_windows(video_id: str, scored: list[ScoredWindow]) -> PlaceabilityReport:
    kept = [window for window in scored if window.kept]
    rejected = [
        RejectedWindow(
            start_s=window.start_s,
            end_s=window.end_s,
            surface_label=window.surface_label,
            reason=window.drop_reason or "Dropped",
        )
        for window in scored
        if not window.kept
    ]
    if not kept:
        reasons: list[str] = []
        for window in scored:
            if window.drop_reason and window.drop_reason not in reasons:
                reasons.append(window.drop_reason)
        return PlaceabilityReport(
            video_id=video_id,
            score=0,
            status="stopped",
            reasons=reasons[:2] or ["No placeable surface"],
            windows=[],
            rejected=rejected,
        )
    best = max(kept, key=lambda window: window.score)
    status = status_for(best.score)
    windows = [
        PlacementWindow(
            start_s=window.start_s,
            end_s=window.end_s,
            surface_label=window.surface_label,
            sam_prompt=window.sam_prompt,
            window_score=window.score,
            signal_scores=window.signals,
            aspect_w=window.aspect_w,
            aspect_h=window.aspect_h,
        )
        for window in sorted(kept, key=lambda window: window.start_s)
    ]
    return PlaceabilityReport(
        video_id=video_id,
        score=best.score,
        status=status,
        reasons=_reasons(best),
        windows=windows,
        rejected=rejected,
    )


def status_for(score: float) -> PlaceabilityStatus:
    if score >= PLACEABLE_SCORE:
        return "placeable"
    if score >= WARNING_SCORE:
        return "warning"
    return "stopped"


def measure_window(scan: SignalScan, start_s: float, end_s: float) -> tuple[float, float]:
    chosen = [sample for sample in scan.samples if start_s <= sample.time_s < end_s]
    flows = [sample.flow for sample in chosen]
    luminances = [sample.luminance for sample in chosen]
    flow = float(sorted(flows)[len(flows) // 2]) if flows else 0.0
    if len(luminances) < 2:
        return flow, 0.0
    mean = sum(luminances) / len(luminances)
    variance = sum((value - mean) ** 2 for value in luminances) / len(luminances)
    return flow, variance


MAX_SPOT_WINDOWS = 3


def assemble_windows(
    plan: ScenePlan,
    scan: SignalScan,
    spots: SpotReport,
    duration_s: float,
) -> list[WindowInput]:
    lookup = {
        (_key(item.start_s), _key(item.end_s), item.sam_prompt): item
        for item in spots.measurements
    }
    windows: list[WindowInput] = []
    for start_s, end_s, label, prompt in placement_spans(plan, scan, duration_s):
        flow, variance = measure_window(scan, start_s, end_s)
        spot = lookup.get((_key(start_s), _key(end_s), prompt))
        windows.append(
            WindowInput(
                start_s=start_s,
                end_s=end_s,
                surface_label=label,
                sam_prompt=prompt,
                flow=flow,
                luminance_variance=variance,
                occlusion=0.0 if spot is None or spot.occlusion is None else spot.occlusion,
                has_cut=False,
                aspect_w=0 if spot is None or spot.aspect_w is None else spot.aspect_w,
                aspect_h=0 if spot is None or spot.aspect_h is None else spot.aspect_h,
                surface_found=False if spot is None else spot.surface_found,
            )
        )
    return windows


def placement_spans(
    plan: ScenePlan,
    scan: SignalScan,
    duration_s: float,
) -> list[tuple[float, float, str, str]]:
    """The stretches worth a SAM spot-check: at most 15 s, and at most three of them."""
    avoids = [(moment.start_s, moment.end_s) for moment in plan.moments_to_avoid]
    ranked: list[tuple[float, float, float, str, str]] = []
    for proposed in plan.windows:
        clipped = subtract_intervals(max(0.0, proposed.start_s), min(duration_s, proposed.end_s), avoids)
        for start_s, end_s in clipped:
            for piece_start, piece_end in split_on_cuts(start_s, end_s, scan.cut_times):
                fitted_start, fitted_end = fit_render_span(piece_start, piece_end, scan)
                flow, _variance = measure_window(scan, fitted_start, fitted_end)
                ranked.append((flow, fitted_start, fitted_end, proposed.surface_label, proposed.sam_prompt))
    ranked.sort(key=lambda item: (item[0], item[1], item[4]))
    chosen = sorted(ranked[:MAX_SPOT_WINDOWS], key=lambda item: (item[1], item[4]))
    return [(start_s, end_s, label, prompt) for _flow, start_s, end_s, label, prompt in chosen]


def fit_render_span(start: float, end: float, scan: SignalScan) -> tuple[float, float]:
    if end - start <= MAX_RENDER_S:
        return start, end
    last = end - MAX_RENDER_S
    cursor = start
    best_start = start
    best_flow = float("inf")
    while True:
        flow = _mean_flow(scan, cursor, cursor + MAX_RENDER_S)
        if flow < best_flow:
            best_flow = flow
            best_start = cursor
        if cursor >= last - 1e-6:
            break
        cursor = min(cursor + 1.0, last)
    return best_start, best_start + MAX_RENDER_S


def _mean_flow(scan: SignalScan, start: float, end: float) -> float:
    flows = [sample.flow for sample in scan.samples if start <= sample.time_s < end]
    if not flows:
        return 0.0
    return sum(flows) / len(flows)


def _key(value: float) -> float:
    return round(value, 3)


def _scored(
    window: WindowInput,
    *,
    kept: bool,
    score: float,
    signals: SignalScores,
    drop_reason: str | None,
) -> ScoredWindow:
    return ScoredWindow(
        kept=kept,
        drop_reason=drop_reason,
        score=score,
        signals=signals,
        start_s=window.start_s,
        end_s=window.end_s,
        surface_label=window.surface_label,
        sam_prompt=window.sam_prompt,
        aspect_w=window.aspect_w,
        aspect_h=window.aspect_h,
        occlusion_fraction=window.occlusion,
    )


def _drop_reason(window: WindowInput, duration: float) -> str | None:
    if duration < MIN_WINDOW_S:
        return "The surface is on screen for under 3 s"
    if not window.surface_found:
        return "No surface visible for 3 s"
    if window.flow > FLOW_LIMIT:
        return "The camera is too shaky"
    if window.occlusion > OCCLUSION_DROP:
        return "The surface is covered more than 80% of the time"
    return None


def _surface_signal(duration: float) -> float:
    return min(max(duration, 0.0), MAX_WINDOW_SCORE_S) / MAX_WINDOW_SCORE_S


def _motion_signal(flow: float) -> float:
    return max(0.0, 1.0 - flow / FLOW_LIMIT)


def _lighting_signal(variance: float) -> float:
    return max(0.0, 1.0 - variance / LIGHTING_VARIANCE_LIMIT)


def _occlusion_signal(fraction: float) -> float:
    if fraction <= 0.10:
        return 0.5 + 0.5 * (fraction / 0.10)
    if fraction <= 0.30:
        return 1.0
    if fraction <= 0.60:
        return 1.0 + (0.4 - 1.0) * ((fraction - 0.30) / 0.30)
    if fraction <= 0.80:
        return 0.4 * (1.0 - (fraction - 0.60) / 0.20)
    return 0.0


def _reasons(window: ScoredWindow) -> list[str]:
    names: tuple[SignalName, ...] = (
        "surface_time",
        "camera_motion",
        "occlusion",
        "lighting",
        "shot_stability",
    )
    ranked = sorted(names, key=lambda name: (getattr(window.signals, name), name))
    return [_reason(name, float(getattr(window.signals, name)), window.occlusion_fraction) for name in ranked[:2]]


def _reason(name: SignalName, value: float, occlusion: float) -> str:
    weak = value < 0.999
    match name:
        case "surface_time":
            if weak:
                return "The surface is not on screen for long enough"
            return "The surface stays on screen"
        case "camera_motion":
            if weak:
                return "The camera moves too much for a stable placement"
            return "The camera is steady"
        case "occlusion":
            if not weak:
                return "People pass in front of the surface often enough to sell the placement"
            if occlusion < 0.10:
                return "Nothing passes in front of the surface, so the placement will look pasted on"
            return "The surface is covered too often"
        case "lighting":
            if weak:
                return "The light on the surface changes too much"
            return "The light on the surface stays even"
        case "shot_stability":
            if weak:
                return "The shot cuts inside this window"
            return "The shot holds"
        case _ as unknown:
            never: Never = unknown
            raise AssertionError(never)
