"""Data interfaces for MiniOneRec JAX scaffold."""

from .artifact_manifest import (
    ARTIFACT_MANIFEST_JSON_SCHEMA,
    ARTIFACT_MANIFEST_SCHEMA_VERSION,
    ArtifactManifest,
    load_manifest,
    save_manifest,
    validate_manifest_dict,
)

__all__ = [
    "ARTIFACT_MANIFEST_JSON_SCHEMA",
    "ARTIFACT_MANIFEST_SCHEMA_VERSION",
    "ArtifactManifest",
    "load_manifest",
    "save_manifest",
    "validate_manifest_dict",
]
