"""CLI entrypoints for the MiniOneRec JAX scaffold."""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings
from typing import Callable, Sequence

import numpy as np

from minionerec_jax.beam_constraints import (
    ConstrainedLogitsProcessor,
    build_prefix_allowed_token_map,
    build_prefix_allowed_tokens_fn,
    hash_tokens,
)
from minionerec_jax.checkpoint import (
    download_checkpoint_snapshot,
    load_easydel_from_torch_checkpoint,
)
from minionerec_jax.config import ProjectConfig, default_config
from minionerec_jax.data.artifact_manifest import (
    ARTIFACT_MANIFEST_JSON_SCHEMA,
    ArtifactManifest,
    save_manifest,
)
from minionerec_jax.eval_metrics import (
    TOPK_LIST,
    build_dry_run_fixture,
    compute_offline_metrics,
    format_metric_lines,
    load_item_name_dict,
    load_predictions_json,
    resolve_item_info_path,
)
from minionerec_jax.logging_utils import configure_logging
from minionerec_jax.paths import get_project_paths


def _cmd_init_manifest(args: argparse.Namespace, config: ProjectConfig) -> int:
    manifest = ArtifactManifest.from_config(config)
    output_path = save_manifest(manifest, args.output)
    print(f"manifest_written={output_path}")
    print(f"schema_version={manifest.schema_version}")
    return 0


def _cmd_print_config(args: argparse.Namespace, config: ProjectConfig) -> int:
    if args.format == "repr":
        print(config)
        return 0

    print(config.to_json())
    return 0


def _cmd_smoke(_: argparse.Namespace, config: ProjectConfig) -> int:
    paths = get_project_paths()
    paths.ensure_artifacts_dir()
    manifest = ArtifactManifest.from_config(config)

    print("smoke_status=ok")
    print(f"project_root={paths.root}")
    print(f"checkpoint_subfolder={config.checkpoint_subfolder}")
    print(f"official_repo_commit={config.official_repo_commit}")
    print(f"hf_model_sha={config.hf_model_sha}")
    print(f"manifest_schema_properties={len(ARTIFACT_MANIFEST_JSON_SCHEMA['properties'])}")
    print(
        "notes=smoke checks scaffold wiring; constrained-mask component is implemented "
        "but full generation loop remains out-of-scope."
    )
    print(f"manifest_preview_checkpoint_path={manifest.checkpoint_path}")
    return 0


def _print_warnings(warnings: Sequence[str]) -> None:
    for warning in warnings:
        print(f"warning={warning}")


def _cmd_download_checkpoint(args: argparse.Namespace, config: ProjectConfig) -> int:
    selected_patterns = (
        tuple(args.allow_pattern)
        if args.allow_pattern
        else config.default_checkpoint_allow_patterns()
    )
    result = download_checkpoint_snapshot(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=args.local_dir,
        checkpoint_subfolder=args.checkpoint_subfolder,
        allow_patterns=selected_patterns,
        dry_run=args.dry_run,
    )

    print("download_checkpoint_status=ok")
    print(f"dry_run={result.dry_run}")
    print(f"repo_id={result.repo_id}")
    print(f"revision={result.revision}")
    print(f"local_dir={result.local_dir}")
    print(f"checkpoint_dir={result.checkpoint_dir}")
    print(f"allow_patterns={','.join(result.allow_patterns)}")
    print(f"huggingface_hub_available={result.huggingface_hub_available}")
    print(
        "notes=use local checkpoint subfolder path for EasyDeL from_torch load "
        "(HF subfolder arg is unreliable)."
    )
    _print_warnings(result.warnings)
    return 0


def _cmd_probe_load(args: argparse.Namespace, _: ProjectConfig) -> int:
    result = load_easydel_from_torch_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        dry_run=args.dry_run,
    )

    print("probe_load_status=ok")
    print(f"dry_run={result.dry_run}")
    print(f"checkpoint_dir={result.checkpoint_dir}")
    print(f"checkpoint_dir_exists={result.checkpoint_dir_exists}")
    print(f"easydel_available={result.easydel_available}")
    print(f"jax_available={result.jax_available}")
    print(f"model_class={result.model_class}")
    print(f"model_type={result.model_type}")
    print(
        "notes=load probe validates conversion path; generation parity remains "
        "a separate unstable track."
    )
    _print_warnings(result.warnings)
    return 0


def _cmd_probe_constraint_mask(args: argparse.Namespace, _: ProjectConfig) -> int:
    eos_token_id = int(args.eos_token_id)

    if "gpt2" in args.base_model.lower():
        tokenized_candidates = ((1, 2, 3, 4, 6), (1, 2, 3, 4, 7))
        step0_input_ids = np.asarray([[1, 2, 3, 4]], dtype=np.int64)
    else:
        tokenized_candidates = ((1, 2, 3, 6), (1, 2, 3, 7))
        step0_input_ids = np.asarray([[0, 1, 2, 3]], dtype=np.int64)

    prefix_map = build_prefix_allowed_token_map(
        tokenized_candidates=tokenized_candidates,
        eos_token_id=eos_token_id,
        base_model=args.base_model,
    )
    prefix_allowed_tokens_fn = build_prefix_allowed_tokens_fn(prefix_map)
    processor = ConstrainedLogitsProcessor(
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        num_beams=1,
        base_model=args.base_model,
        eos_token_id=eos_token_id,
    )

    step0_scores = np.asarray([[0.1, 0.0, 0.2, 0.3, 0.1, 0.4, 1.2, 1.1, 0.2, 0.0]], dtype=np.float64)
    step0_output = np.asarray(processor(step0_input_ids, step0_scores), dtype=np.float64)
    step0_allowed = np.where(np.isfinite(step0_output[0]))[0].tolist()
    count_after_step0 = processor.count

    step1_input_ids = np.concatenate(
        [step0_input_ids, np.asarray([[8]], dtype=np.int64)],
        axis=1,
    )
    step1_scores = np.asarray([[0.3, 0.1, 0.0, 0.4, 0.2, 0.5, 1.0, 0.2, 1.1, 0.0]], dtype=np.float64)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        step1_output = np.asarray(processor(step1_input_ids, step1_scores), dtype=np.float64)
    step1_allowed = np.where(np.isfinite(step1_output[0]))[0].tolist()

    print("probe_constraint_mask_status=ok")
    print(f"base_model={args.base_model}")
    print(f"prefix_index={processor.prefix_index}")
    print(f"count_after_step0={count_after_step0}")
    print(f"count_after_step1={processor.count}")
    print(f"step0_prefix_key={hash_tokens(step0_input_ids[0, -processor.prefix_index:].tolist())}")
    print(f"step1_prefix_key={hash_tokens(step1_input_ids[0, -count_after_step0:].tolist())}")
    print(f"step0_allowed_tokens={','.join(str(token) for token in step0_allowed)}")
    print(f"step1_allowed_tokens={','.join(str(token) for token in step1_allowed)}")
    print(f"eos_token_id={eos_token_id}")
    print(f"step1_warned={len(caught) > 0}")
    print(f"step1_eos_fallback={step1_allowed == [eos_token_id]}")
    return 0


def _cmd_eval_metrics(args: argparse.Namespace, _: ProjectConfig) -> int:
    if args.dry_run:
        prediction_rows, item_name_dict = build_dry_run_fixture()
        mode = "dry-run"
    else:
        if args.predictions_json is None or args.item_info is None:
            raise RuntimeError(
                "`eval-metrics` requires `--predictions-json` and `--item-info` unless "
                "`--dry-run` is set."
            )
        prediction_rows = load_predictions_json(args.predictions_json)
        item_name_dict = load_item_name_dict(args.item_info)
        mode = "files"

    result = compute_offline_metrics(
        prediction_rows,
        topk_list=TOPK_LIST,
        item_name_dict=item_name_dict,
    )

    print("eval_metrics_status=ok")
    print(f"mode={mode}")
    if not args.dry_run:
        assert args.predictions_json is not None
        assert args.item_info is not None
        print(f"predictions_json={Path(args.predictions_json).expanduser().resolve()}")
        print(f"item_info_txt={resolve_item_info_path(args.item_info)}")
    print(f"sample_count={result.sample_count}")
    print(f"n_beam={result.n_beam}")
    print(f"valid_topk={','.join(str(k) for k in result.valid_topk)}")
    print(f"missing_item_predictions={result.missing_item_predictions}")
    for line in format_metric_lines(result):
        print(line)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="minionerec-jax",
        description="Scaffold CLI for MiniOneRec JAX reproduction work.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (default: INFO).",
    )

    cfg = default_config()
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_manifest_parser = subparsers.add_parser(
        "init-manifest",
        help="Write an artifact manifest JSON file from default pinned config.",
    )
    init_manifest_parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/manifest.json"),
        help="Output path for artifact manifest.",
    )
    init_manifest_parser.set_defaults(handler=_cmd_init_manifest)

    print_config_parser = subparsers.add_parser(
        "print-config",
        help="Print the scaffold configuration with pinned artifacts.",
    )
    print_config_parser.add_argument(
        "--format",
        choices=("json", "repr"),
        default="json",
        help="Output format.",
    )
    print_config_parser.set_defaults(handler=_cmd_print_config)

    smoke_parser = subparsers.add_parser(
        "smoke",
        help="Run placeholder scaffold checks without model generation.",
    )
    smoke_parser.set_defaults(handler=_cmd_smoke)

    download_checkpoint_parser = subparsers.add_parser(
        "download-checkpoint",
        help="Download pinned checkpoint artifacts with allow-pattern filtering.",
    )
    download_checkpoint_parser.add_argument(
        "--repo-id",
        default=cfg.hf_repo_id,
        help="Hugging Face repository id.",
    )
    download_checkpoint_parser.add_argument(
        "--revision",
        default=cfg.hf_model_sha,
        help="Repository revision to pin (default: configured model sha).",
    )
    download_checkpoint_parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path(cfg.hf_snapshot_local_dir),
        help="Local snapshot destination directory.",
    )
    download_checkpoint_parser.add_argument(
        "--checkpoint-subfolder",
        default=cfg.checkpoint_subfolder,
        help="Checkpoint subfolder name used to report the local load path.",
    )
    download_checkpoint_parser.add_argument(
        "--allow-pattern",
        action="append",
        default=[],
        help=(
            "Pattern to pass to snapshot_download allow_patterns. "
            "Provide multiple times for multiple patterns."
        ),
    )
    download_checkpoint_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate arguments and dependency visibility without downloading.",
    )
    download_checkpoint_parser.set_defaults(handler=_cmd_download_checkpoint)

    probe_load_parser = subparsers.add_parser(
        "probe-load",
        help="Probe EasyDeL from_torch load from a local checkpoint subfolder.",
    )
    probe_load_parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Path to local checkpoint subfolder, e.g. artifacts/hf_snapshot/Industrial_ckpt.",
    )
    probe_load_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate path/dependencies only without importing/loading EasyDeL model weights.",
    )
    probe_load_parser.set_defaults(handler=_cmd_probe_load)

    probe_constraint_mask_parser = subparsers.add_parser(
        "probe-constraint-mask",
        help="Run deterministic constrained-logits masking diagnostics.",
    )
    probe_constraint_mask_parser.add_argument(
        "--base-model",
        default="gpt2",
        help="Model name used to select prefix index semantics (gpt2 => 4 else 3).",
    )
    probe_constraint_mask_parser.add_argument(
        "--eos-token-id",
        type=int,
        default=9,
        help="EOS token id used for fallback when no allowed tokens exist.",
    )
    probe_constraint_mask_parser.set_defaults(handler=_cmd_probe_constraint_mask)

    eval_metrics_parser = subparsers.add_parser(
        "eval-metrics",
        help="Compute MiniOneRec offline metrics (HR@K/NDCG@K) from prediction JSON.",
    )
    eval_metrics_parser.add_argument(
        "--predictions-json",
        type=Path,
        default=None,
        help="Path to generated prediction JSON with `predict` and `output` fields.",
    )
    eval_metrics_parser.add_argument(
        "--item-info",
        type=Path,
        default=None,
        help="Item info text path (with or without .txt extension).",
    )
    eval_metrics_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run with embedded tiny fixture (no external files required).",
    )
    eval_metrics_parser.set_defaults(handler=_cmd_eval_metrics)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    config = default_config()
    handler: Callable[[argparse.Namespace, ProjectConfig], int] = args.handler
    try:
        return handler(args, config)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"error={exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
