"""Configuration objects for the MiniOneRec JAX scaffold."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any


@dataclass(frozen=True)
class ProjectConfig:
    """Single-source configuration for pins and artifact locations."""

    official_repo_commit: str = "8e03e354033fc81f830580f01c102bd7fbaa262a"
    hf_repo_id: str = "kkknight/MiniOneRec"
    hf_model_sha: str = "365a03fc6601ed36abd69cc9a0a59025a3d31cdc"
    checkpoint_subfolder: str = "Industrial_ckpt"
    hf_snapshot_local_dir: str = "artifacts/hf_snapshot"
    dataset_train_path: str = "artifacts/datasets/Industrial_and_Scientific/train.jsonl"
    dataset_validation_path: str = "artifacts/datasets/Industrial_and_Scientific/valid.jsonl"
    dataset_test_path: str = "artifacts/datasets/Industrial_and_Scientific/test.jsonl"
    artifact_manifest_path: str = "artifacts/manifest.json"
    eval_output_dir: str = "artifacts/eval"
    log_level: str = "INFO"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def with_overrides(self, **overrides: Any) -> "ProjectConfig":
        merged = self.to_dict()
        merged.update(overrides)
        return ProjectConfig(**merged)

    def default_checkpoint_allow_patterns(self) -> tuple[str, ...]:
        return (f"{self.checkpoint_subfolder}/*",)


def default_config() -> ProjectConfig:
    return ProjectConfig()
