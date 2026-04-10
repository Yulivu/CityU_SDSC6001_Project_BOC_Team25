from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def outputs_dir() -> Path:
    return project_root() / "outputs"


def artifacts_dir() -> Path:
    d = project_root() / "artifacts"
    d.mkdir(parents=True, exist_ok=True)
    return d

