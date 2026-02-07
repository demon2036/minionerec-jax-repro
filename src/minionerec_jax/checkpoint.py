"""Checkpoint download and load helpers for MiniOneRec JAX."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class DownloadCheckpointResult:
    dry_run: bool
    repo_id: str
    revision: str | None
    local_dir: Path
    checkpoint_dir: Path
    allow_patterns: tuple[str, ...]
    huggingface_hub_available: bool
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProbeLoadResult:
    dry_run: bool
    checkpoint_dir: Path
    checkpoint_dir_exists: bool
    easydel_available: bool
    jax_available: bool
    model_class: str | None = None
    model_type: str | None = None
    warnings: tuple[str, ...] = ()


def resolve_local_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def resolve_checkpoint_subfolder_dir(
    snapshot_root: str | Path,
    checkpoint_subfolder: str,
    *,
    must_exist: bool,
) -> Path:
    root = resolve_local_path(snapshot_root)
    checkpoint_dir = (root / checkpoint_subfolder).resolve()
    if must_exist and not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            "Checkpoint directory does not exist: "
            f"{checkpoint_dir}. Download the checkpoint first with `download-checkpoint`."
        )
    return checkpoint_dir


def _import_snapshot_download(*, required: bool) -> tuple[Callable[..., str] | None, str | None]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        message = (
            "Missing optional dependency `huggingface_hub`. "
            "Install with `pip install huggingface-hub` to enable checkpoint downloads."
        )
        if required:
            raise RuntimeError(message) from None
        return None, message
    return snapshot_download, None


def download_checkpoint_snapshot(
    *,
    repo_id: str,
    revision: str | None,
    local_dir: str | Path,
    checkpoint_subfolder: str,
    allow_patterns: Sequence[str],
    dry_run: bool,
) -> DownloadCheckpointResult:
    local_dir_path = resolve_local_path(local_dir)
    patterns = tuple(allow_patterns)
    warnings: list[str] = []

    snapshot_download, dependency_warning = _import_snapshot_download(required=not dry_run)
    hf_available = snapshot_download is not None
    if dependency_warning is not None:
        warnings.append(dependency_warning)

    if not dry_run:
        assert snapshot_download is not None
        downloaded_dir = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(local_dir_path),
            allow_patterns=list(patterns),
        )
        local_dir_path = resolve_local_path(downloaded_dir)

    checkpoint_dir = resolve_checkpoint_subfolder_dir(
        local_dir_path,
        checkpoint_subfolder,
        must_exist=False,
    )

    return DownloadCheckpointResult(
        dry_run=dry_run,
        repo_id=repo_id,
        revision=revision,
        local_dir=local_dir_path,
        checkpoint_dir=checkpoint_dir,
        allow_patterns=patterns,
        huggingface_hub_available=hf_available,
        warnings=tuple(warnings),
    )


def _module_available(module_name: str) -> bool:
    try:
        __import__(module_name)
    except ImportError:
        return False
    return True


def _extract_model_type(model: Any) -> str | None:
    model_config = getattr(model, "config", None)
    if model_config is None:
        return None
    return getattr(model_config, "model_type", None)


def apply_generation_config_compat_stub(model: Any) -> None:
    """Future hook for generation-config compatibility patches."""

    _ = model


def load_easydel_from_torch_checkpoint(
    checkpoint_dir: str | Path,
    *,
    dry_run: bool,
) -> ProbeLoadResult:
    resolved_checkpoint_dir = resolve_local_path(checkpoint_dir)
    checkpoint_exists = resolved_checkpoint_dir.is_dir()

    easydel_available = _module_available("easydel")
    jax_available = _module_available("jax")

    warnings: list[str] = []
    if not checkpoint_exists:
        warnings.append(
            "Checkpoint directory is missing. Expected a local subfolder path from a snapshot "
            f"download, e.g. `{resolved_checkpoint_dir}`."
        )
    if not easydel_available:
        warnings.append(
            "Optional dependency `easydel` is missing. Install with `pip install easydel` "
            "for non-dry-run load probes."
        )
    if not jax_available:
        warnings.append(
            "Optional dependency `jax` is missing. Install with `pip install jax` "
            "for non-dry-run load probes."
        )

    if dry_run:
        return ProbeLoadResult(
            dry_run=True,
            checkpoint_dir=resolved_checkpoint_dir,
            checkpoint_dir_exists=checkpoint_exists,
            easydel_available=easydel_available,
            jax_available=jax_available,
            warnings=tuple(warnings),
        )

    if not checkpoint_exists:
        raise FileNotFoundError(
            "Checkpoint directory does not exist: "
            f"{resolved_checkpoint_dir}. Run `download-checkpoint` and pass the local subfolder "
            "path to `probe-load --checkpoint-dir`."
        )

    missing_runtime = []
    if not easydel_available:
        missing_runtime.append("easydel")
    if not jax_available:
        missing_runtime.append("jax")
    if missing_runtime:
        missing_csv = ", ".join(missing_runtime)
        raise RuntimeError(
            "`probe-load` requires runtime dependencies for non-dry-run mode. "
            f"Missing: {missing_csv}. Install with `pip install easydel jax`."
        )

    try:
        from easydel import AutoEasyDeLModelForCausalLM
    except Exception as exc:
        raise RuntimeError(
            "Failed to import `AutoEasyDeLModelForCausalLM` from `easydel`. "
            "Check `easydel` and `jax` installation/version compatibility."
        ) from exc

    try:
        model = AutoEasyDeLModelForCausalLM.from_pretrained(
            str(resolved_checkpoint_dir),
            from_torch=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "EasyDeL `from_torch` load failed for local checkpoint directory "
            f"`{resolved_checkpoint_dir}`. Ensure this path points to the downloaded checkpoint "
            "subfolder and not a remote HF subfolder reference."
        ) from exc

    apply_generation_config_compat_stub(model)

    return ProbeLoadResult(
        dry_run=False,
        checkpoint_dir=resolved_checkpoint_dir,
        checkpoint_dir_exists=checkpoint_exists,
        easydel_available=easydel_available,
        jax_available=jax_available,
        model_class=model.__class__.__name__,
        model_type=_extract_model_type(model),
        warnings=tuple(warnings),
    )
