from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from pathlib import Path

from inplace.errors import PipelineError, VideoRejected
from inplace.models import StageManifest


def cache_key(*, stage: str, version: str, inputs: dict[str, str]) -> str:
    payload = json.dumps(
        {"stage": stage, "version": version, "inputs": inputs},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class JobStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    @property
    def id(self) -> str:
        return self.root.name

    @classmethod
    def create(cls, jobs_root: Path, source: Path) -> JobStore:
        if source.suffix.lower() != ".mp4":
            raise VideoRejected("Only MP4 uploads are accepted")
        if not source.is_file():
            raise VideoRejected(f"No video at {source}")
        root = Path(jobs_root) / uuid.uuid4().hex[:12]
        root.mkdir(parents=True)
        shutil.copy(source, root / "source.mp4")
        return cls(root)

    @classmethod
    def open(cls, jobs_root: Path, job: str) -> JobStore:
        candidate = Path(job)
        root = candidate if candidate.is_dir() else Path(jobs_root) / job
        if not (root / "source.mp4").is_file():
            raise PipelineError(f"No job at {root}")
        return cls(root)

    def path(self, relative: str) -> Path:
        return self.root / relative

    def stage_dir(self, stage: str) -> Path:
        return self.root / "stages" / stage

    def read_manifest(self, stage: str) -> StageManifest | None:
        manifest_path = self.stage_dir(stage) / "manifest.json"
        if not manifest_path.is_file():
            return None
        return StageManifest.model_validate_json(manifest_path.read_text())

    def write_manifest(self, *, stage: str, cache_key: str, version: str, outputs: list[str]) -> None:
        directory = self.stage_dir(stage)
        directory.mkdir(parents=True, exist_ok=True)
        manifest = StageManifest(stage=stage, version=version, cache_key=cache_key, outputs=outputs)
        (directory / "manifest.json").write_text(manifest.model_dump_json(indent=2))

    def write_error(self, stage: str, message: str) -> None:
        directory = self.stage_dir(stage)
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
        payload = {"status": "error", "message": message}
        (directory / "error.json").write_text(json.dumps(payload, indent=2))

    def is_cached(self, stage: str, key: str, outputs: list[str]) -> bool:
        manifest = self.read_manifest(stage)
        if manifest is None or manifest.status != "ok" or manifest.cache_key != key:
            return False
        return all(self.path(relative).is_file() for relative in outputs)

    def remove(self, path: Path) -> None:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.is_file() or path.is_symlink():
            path.unlink()

    def clear_derived(self, stage: str) -> None:
        manifest = self.read_manifest(stage)
        if manifest is not None:
            for relative in manifest.outputs:
                self.remove(self.path(relative))
        directory = self.stage_dir(stage)
        if directory.exists():
            shutil.rmtree(directory)

    def write_model(self, relative: str, model: object) -> None:
        target = self.path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        dump = getattr(model, "model_dump_json")
        target.write_text(dump(indent=2))
