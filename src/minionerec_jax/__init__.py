"""MiniOneRec JAX scaffold package."""

from .beam_constraints import (
    ConstrainedLogitsProcessor,
    build_prefix_allowed_token_map,
    build_prefix_allowed_tokens_fn,
    get_prefix_index,
    hash_tokens,
)
from .config import ProjectConfig, default_config
from .eval_metrics import (
    TOPK_LIST,
    OfflineMetricsResult,
    compute_offline_metrics,
    extract_target_item,
    format_metric_lines,
)

__all__ = [
    "ConstrainedLogitsProcessor",
    "OfflineMetricsResult",
    "ProjectConfig",
    "TOPK_LIST",
    "build_prefix_allowed_token_map",
    "build_prefix_allowed_tokens_fn",
    "compute_offline_metrics",
    "default_config",
    "extract_target_item",
    "format_metric_lines",
    "get_prefix_index",
    "hash_tokens",
]

__version__ = "0.1.0"
