from __future__ import annotations

import shutil

from inplace.constants import IMAGE_MODEL
from inplace.errors import PipelineError
from inplace.jobs import JobStore, hash_file
from inplace.meta_client import MetaClient
from inplace.models import BrandBrief, Placement
from inplace.runner import run_cached

CREATIVE_VERSION = "1"

SAMPLE_BRANDS = {
    "northline": BrandBrief(
        name="Northline",
        description="A fictional trail-food brand.",
        style="flat geometric poster, deep green, cream type",
    ),
    "lumen soda": BrandBrief(
        name="Lumen Soda",
        description="A fictional citrus soda.",
        style="flat poster, yellow and navy, bold type",
    ),
    "paper orbit": BrandBrief(
        name="Paper Orbit",
        description="A fictional stationery studio.",
        style="flat poster, warm paper tones, simple wordmark",
    ),
}


def resolve_brand(name: str, description: str, style: str) -> BrandBrief:
    sample = SAMPLE_BRANDS.get(name.strip().lower())
    if sample is not None and not description.strip() and not style.strip():
        return sample
    if sample is not None and not description.strip():
        return BrandBrief(name=sample.name, description=sample.description, style=style.strip() or sample.style)
    if not description.strip() or not style.strip():
        known = ", ".join(sorted(SAMPLE_BRANDS))
        raise PipelineError(
            f"A brand needs a one-line description and a style hint, or use a sample brand: {known}"
        )
    return BrandBrief(name=name.strip(), description=description.strip(), style=style.strip())


# Muse Image accepts these canvases. A custom size such as 748x1536 is rejected.
_CANVASES = ((1024, 1024), (1536, 1024), (1024, 1536))


def muse_size(aspect_w: int, aspect_h: int) -> str:
    if aspect_w <= 0 or aspect_h <= 0:
        raise PipelineError("The surface aspect ratio is unknown. Run the placeability check again.")
    target = aspect_w / aspect_h
    width, height = min(_CANVASES, key=lambda canvas: abs((canvas[0] / canvas[1]) - target))
    return f"{width}x{height}"


def run_creative(job: JobStore, client: MetaClient, force: set[str]) -> str:
    placement = Placement.model_validate_json(job.path("placement.json").read_text())
    size = muse_size(placement.aspect_w, placement.aspect_h)
    prompt = _prompt(placement.brand)

    def produce() -> None:
        for index in range(3):
            paid = job.path(f"paid/creative/{index}.png")
            if not paid.is_file() or paid.stat().st_size == 0:
                paid.parent.mkdir(parents=True, exist_ok=True)
                paid.write_bytes(client.generate_image(prompt=prompt, size=size))
        for index in range(3):
            target = job.path(f"creative/{index}.png")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(job.path(f"paid/creative/{index}.png"), target)

    return run_cached(
        job,
        stage="creative",
        version=CREATIVE_VERSION,
        inputs={
            "brand": placement.brand.name,
            "description": placement.brand.description,
            "style": placement.brand.style,
            "aspect": size,
            "model": IMAGE_MODEL,
        },
        outputs=["creative/0.png", "creative/1.png", "creative/2.png"],
        force="creative" in force,
        fn=produce,
    )


def _prompt(brand: BrandBrief) -> str:
    return (
        "Flat printed poster artwork for a fictional brand. "
        "Do not depict a real company, a real product, or a real logo. "
        f"Brand name: {brand.name}. "
        f"Description: {brand.description} "
        f"Style: {brand.style}. "
        "The poster fills the frame edge to edge. No room, no mockup, no people."
    )
