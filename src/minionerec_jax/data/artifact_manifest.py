"""Artifact manifest schema and serialization helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

from minionerec_jax.config import ProjectConfig

ARTIFACT_MANIFEST_SCHEMA_VERSION = "1.0.0"

ARTIFACT_MANIFEST_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "MiniOneRecJaxArtifactManifest",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "official_repo_commit",
        "hf_repo_id",
        "hf_model_sha",
        "checkpoint_subfolder",
        "checkpoint_path",
        "dataset_train_path",
        "dataset_validation_path",
        "dataset_test_path",
        "notes",
    ],
    "properties": {
        "schema_version": {"type": "string"},
        "official_repo_commit": {"type": "string"},
        "hf_repo_id": {"type": "string"},
        "hf_model_sha": {"type": "string"},
        "checkpoint_subfolder": {"type": "string"},
        "checkpoint_path": {"type": "string"},
        "dataset_train_path": {"type": "string"},
        "dataset_validation_path": {"type": "string"},
        "dataset_test_path": {"type": "string"},
        "notes": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}


@dataclass
class ArtifactManifest:
    schema_version: str = ARTIFACT_MANIFEST_SCHEMA_VERSION
    official_repo_commit: str = "8e03e354033fc81f830580f01c102bd7fbaa262a"
    hf_repo_id: str = "kkknight/MiniOneRec"
    hf_model_sha: str = "365a03fc6601ed36abd69cc9a0a59025a3d31cdc"
    checkpoint_subfolder: str = "Industrial_ckpt"
    checkpoint_path: str = "artifacts/checkpoints/Industrial_ckpt"
    dataset_train_path: str = "artifacts/datasets/Industrial_and_Scientific/train.jsonl"
    dataset_validation_path: str = "artifacts/datasets/Industrial_and_Scientific/valid.jsonl"
    dataset_test_path: str = "artifacts/datasets/Industrial_and_Scientific/test.jsonl"
    notes: list[str] = field(default_factory=list)

    @classmethod
    def from_config(cls, config: ProjectConfig) -> "ArtifactManifest":
        return cls(
            official_repo_commit=config.official_repo_commit,
            hf_repo_id=config.hf_repo_id,
            hf_model_sha=config.hf_model_sha,
            checkpoint_subfolder=config.checkpoint_subfolder,
            checkpoint_path=f"artifacts/checkpoints/{config.checkpoint_subfolder}",
            dataset_train_path=config.dataset_train_path,
            dataset_validation_path=config.dataset_validation_path,
            dataset_test_path=config.dataset_test_path,
            notes=[
                "Scaffold manifest only.",
                "Checkpoint conversion and constrained beam generation are pending.",
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def validate_manifest_dict(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("Manifest payload must be a mapping.")

    properties = set(ARTIFACT_MANIFEST_JSON_SCHEMA["properties"].keys())
    required = set(ARTIFACT_MANIFEST_JSON_SCHEMA["required"])

    missing = required - set(payload.keys())
    if missing:
        raise ValueError(f"Manifest missing required keys: {sorted(missing)}")

    extra = set(payload.keys()) - properties
    if extra:
        raise ValueError(f"Manifest contains unsupported keys: {sorted(extra)}")

    if payload["schema_version"] != ARTIFACT_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version={payload['schema_version']}; expected {ARTIFACT_MANIFEST_SCHEMA_VERSION}"
        )

    string_fields = [
        "schema_version",
        "official_repo_commit",
        "hf_repo_id",
        "hf_model_sha",
        "checkpoint_subfolder",
        "checkpoint_path",
        "dataset_train_path",
        "dataset_validation_path",
        "dataset_test_path",
    ]
    for field_name in string_fields:
        if not isinstance(payload[field_name], str):
            raise ValueError(f"Manifest field '{field_name}' must be a string.")

    notes = payload["notes"]
    if not isinstance(notes, list) or not all(isinstance(item, str) for item in notes):
        raise ValueError("Manifest field 'notes' must be a list of strings.")


def save_manifest(manifest: ArtifactManifest, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.to_json() + "\n", encoding="utf-8")
    return path


def load_manifest(input_path: str | Path) -> ArtifactManifest:
    path = Path(input_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest_dict(payload)
    return ArtifactManifest(**payload)
