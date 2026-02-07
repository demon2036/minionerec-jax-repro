"""Offline metrics for MiniOneRec-style recommendation outputs."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


TOPK_LIST: tuple[int, ...] = (1, 3, 5, 10, 20, 50)


@dataclass(frozen=True)
class OfflineMetricsResult:
    sample_count: int
    n_beam: int
    valid_topk: tuple[int, ...]
    hr: dict[int, float]
    ndcg: dict[int, float]
    missing_item_predictions: int


def _normalize_prediction_text(value: Any) -> str:
    return str(value).strip("\"\n").strip()


def extract_target_item(output_value: Any) -> str:
    if isinstance(output_value, list):
        if not output_value:
            return ""
        return str(output_value[0]).strip("\"").strip(" ")
    return str(output_value).strip(" \n\"")


def resolve_item_info_path(item_info_path: str | Path) -> Path:
    path = str(item_info_path)
    if path.endswith(".txt"):
        path = path[:-4]
    return Path(f"{path}.txt").expanduser().resolve()


def build_item_name_dict(item_names: Sequence[str]) -> dict[str, list[int]]:
    item_dict: dict[str, list[int]] = {}
    for index, item_name in enumerate(item_names):
        key = item_name.strip()
        if key not in item_dict:
            item_dict[key] = [index]
        else:
            item_dict[key].append(index)
    return item_dict


def load_item_name_dict(item_info_path: str | Path) -> dict[str, list[int]]:
    resolved_path = resolve_item_info_path(item_info_path)
    with resolved_path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()

    item_names = [line.split("\t")[0].strip() for line in lines]
    return build_item_name_dict(item_names)


def load_predictions_json(predictions_json: str | Path) -> list[dict[str, Any]]:
    resolved_path = Path(predictions_json).expanduser().resolve()
    with resolved_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        raise RuntimeError("Predictions JSON must be a list of samples.")

    return payload


def compute_offline_metrics(
    prediction_rows: Sequence[Mapping[str, Any]],
    *,
    topk_list: Sequence[int] = TOPK_LIST,
    item_name_dict: Mapping[str, Sequence[int]] | None = None,
) -> OfflineMetricsResult:
    if not prediction_rows:
        return OfflineMetricsResult(
            sample_count=0,
            n_beam=0,
            valid_topk=(),
            hr={},
            ndcg={},
            missing_item_predictions=0,
        )

    first_predict = prediction_rows[0].get("predict")
    if not isinstance(first_predict, list):
        raise RuntimeError("Each sample must include a list-valued `predict` field.")

    n_beam = len(first_predict)
    valid_topk = tuple(topk for topk in topk_list if topk <= n_beam)
    hr_accumulator = [0.0] * len(topk_list)
    ndcg_accumulator = [0.0] * len(topk_list)
    missing_item_predictions = 0

    for row in prediction_rows:
        if "predict" not in row or "output" not in row:
            raise RuntimeError("Each sample must include both `predict` and `output` fields.")

        raw_predictions = row["predict"]
        if not isinstance(raw_predictions, list):
            raise RuntimeError("Each sample `predict` field must be a list.")

        predictions = [_normalize_prediction_text(value) for value in raw_predictions]
        target_item = extract_target_item(row["output"])

        min_rank = 1_000_000
        for rank, prediction in enumerate(predictions):
            if item_name_dict is not None and prediction not in item_name_dict:
                missing_item_predictions += 1
            if prediction == target_item:
                min_rank = rank
                break

        for index, topk in enumerate(topk_list):
            if topk > n_beam:
                continue
            if min_rank < topk:
                ndcg_accumulator[index] += 1.0 / math.log(min_rank + 2)
                hr_accumulator[index] += 1.0

    sample_count = len(prediction_rows)
    hr = {}
    ndcg = {}
    for index, topk in enumerate(topk_list):
        if topk > n_beam:
            continue
        hr[topk] = hr_accumulator[index] / sample_count
        ndcg[topk] = ndcg_accumulator[index] / sample_count / (1.0 / math.log(2))

    return OfflineMetricsResult(
        sample_count=sample_count,
        n_beam=n_beam,
        valid_topk=valid_topk,
        hr=hr,
        ndcg=ndcg,
        missing_item_predictions=missing_item_predictions,
    )


def format_metric_lines(result: OfflineMetricsResult) -> list[str]:
    lines: list[str] = []
    for topk in result.valid_topk:
        lines.append(f"hr@{topk}={result.hr[topk]:.12f}")
        lines.append(f"ndcg@{topk}={result.ndcg[topk]:.12f}")
    return lines


def build_dry_run_fixture() -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    predictions = [
        {
            "predict": ["item-alpha", "item-beta", "item-gamma", "item-delta", "item-epsilon"],
            "output": "item-alpha",
        },
        {
            "predict": ["item-theta", "item-iota", "item-kappa", "item-lambda", "item-mu"],
            "output": ["item-kappa"],
        },
        {
            "predict": ["item-nu", "item-xi", "item-omicron", "item-pi", "item-rho"],
            "output": "item-sigma",
        },
    ]
    item_dict = build_item_name_dict(
        [
            "item-alpha",
            "item-beta",
            "item-gamma",
            "item-delta",
            "item-epsilon",
            "item-theta",
            "item-iota",
            "item-kappa",
            "item-lambda",
            "item-mu",
            "item-nu",
            "item-xi",
            "item-omicron",
            "item-pi",
            "item-rho",
            "item-sigma",
        ]
    )
    return predictions, item_dict
