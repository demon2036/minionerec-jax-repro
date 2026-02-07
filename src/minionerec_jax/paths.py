"""Filesystem path helpers for the MiniOneRec JAX scaffold."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    src: Path
    artifacts: Path
    scripts: Path

    def ensure_artifacts_dir(self) -> Path:
        self.artifacts.mkdir(parents=True, exist_ok=True)
        return self.artifacts


def resolve_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
            return candidate

    return current


def get_project_paths(start: Path | None = None) -> ProjectPaths:
    root = resolve_project_root(start)
    return ProjectPaths(
        root=root,
        src=root / "src",
        artifacts=root / "artifacts",
        scripts=root / "scripts",
    )
