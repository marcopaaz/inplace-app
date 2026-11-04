import os
from pathlib import Path

from inplace.meta_client import load_project_env


def test_env_file_fills_a_missing_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    (tmp_path / ".env").write_text('MODEL_API_KEY="from-file"\n')
    monkeypatch.chdir(tmp_path)
    load_project_env()
    assert os.environ["MODEL_API_KEY"] == "from-file"


def test_existing_environment_wins_over_the_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MODEL_API_KEY", "from-env")
    (tmp_path / ".env").write_text("MODEL_API_KEY=from-file\n")
    monkeypatch.chdir(tmp_path)
    load_project_env()
    assert os.environ["MODEL_API_KEY"] == "from-env"
