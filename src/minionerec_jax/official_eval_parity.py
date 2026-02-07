"""Official `evaluate.py` parity pipeline for EasyDeL/JAX inference."""

from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np

from minionerec_jax.beam_constraints import (
    ConstrainedLogitsProcessor,
    build_prefix_allowed_token_map,
    build_prefix_allowed_tokens_fn,
)
from minionerec_jax.checkpoint import resolve_local_path


CATEGORY_TEXT_MAP: dict[str, str] = {
    "Industrial_and_Scientific": "industrial and scientific items",
    "Office_Products": "office products",
    "Toys_and_Games": "toys and games",
    "Sports": "sports and outdoors",
    "Books": "books",
}

INSTRUCTION_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Can you predict the next possible item that the user may expect?

"""

USER_PROMPT_TEMPLATE = """### User Input: 
{user_input}

### Response:\n{response}"""


@dataclass(frozen=True)
class OfficialEvalParityResult:
    dry_run: bool
    checkpoint_dir: Path
    info_file: Path
    test_csv: Path
    result_json: Path
    category: str
    category_text: str
    sample_count: int
    written_count: int
    batch_size: int
    num_beams: int
    max_new_tokens: int
    length_penalty: float
    seed: int
    limit: int | None
    sharding_axis_dims: tuple[int, ...]
    easydel_available: bool
    jax_available: bool
    transformers_available: bool
    warnings: tuple[str, ...] = ()


class OfficialTokenizerAdapter:
    """Tokenizer adapter mirroring official data.py bos/eos handling."""

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        self.bos_id = getattr(tokenizer, "bos_token_id", None)
        self.eos_id = getattr(tokenizer, "eos_token_id", None)

    def encode(self, text: str, *, bos: bool, eos: bool) -> list[int]:
        token_ids = list(self.tokenizer.encode(text))

        while token_ids and self.bos_id is not None and token_ids[0] == self.bos_id:
            token_ids = token_ids[1:]
        while token_ids and self.eos_id is not None and token_ids[-1] == self.eos_id:
            token_ids = token_ids[:-1]

        if bos and self.bos_id is not None:
            token_ids = [int(self.bos_id)] + token_ids
        if eos and self.eos_id is not None:
            token_ids = token_ids + [int(self.eos_id)]
        return [int(token) for token in token_ids]


def _module_available(module_name: str) -> bool:
    try:
        __import__(module_name)
    except ImportError:
        return False
    return True


def parse_sharding_axis_dims(raw_value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(raw_value, str):
        values = [part.strip() for part in raw_value.split(",") if part.strip()]
        if not values:
            raise RuntimeError("`--sharding-axis-dims` must not be empty.")
        dims = tuple(int(value) for value in values)
    else:
        dims = tuple(int(value) for value in raw_value)

    if len(dims) not in (4, 5):
        raise RuntimeError(
            "`sharding_axis_dims` must contain 4 or 5 integers, "
            "e.g. `1,1,1,-1` or `1,1,1,-1,1`. "
            f"Received: {dims}"
        )
    return dims


def _resolve_axis_dims_for_device_count(axis_dims: Sequence[int], device_count: int) -> tuple[int, ...]:
    dims = [int(value) for value in axis_dims]
    unresolved_indexes = [index for index, value in enumerate(dims) if value == -1]

    if len(unresolved_indexes) > 1:
        return tuple(dims)

    if len(unresolved_indexes) == 1:
        unresolved_index = unresolved_indexes[0]
        known_product = 1
        for value in dims:
            if value != -1:
                known_product *= abs(value)
        if known_product > 0 and device_count > 0 and device_count % known_product == 0:
            dims[unresolved_index] = device_count // known_product

    return tuple(abs(value) for value in dims)


def _should_retry_mesh_with_4d_axis_dims(error: Exception, dims: tuple[int, ...]) -> bool:
    if len(dims) != 5:
        return False
    message = str(error)
    return "devices.ndim == 5 and len(axis_names) == 4" in message


def resolve_category_text(category: str) -> str:
    if category not in CATEGORY_TEXT_MAP:
        allowed = ", ".join(sorted(CATEGORY_TEXT_MAP.keys()))
        raise RuntimeError(f"Unknown category `{category}`. Allowed categories: {allowed}")
    return CATEGORY_TEXT_MAP[category]


def _extract_input_ids(tokenized: Any) -> list[int]:
    if isinstance(tokenized, Mapping):
        input_ids = tokenized.get("input_ids")
    else:
        input_ids = getattr(tokenized, "input_ids", None)

    if input_ids is None:
        raise RuntimeError("Tokenizer output does not contain `input_ids`.")

    if isinstance(input_ids, np.ndarray):
        input_ids = input_ids.tolist()
    if isinstance(input_ids, tuple):
        input_ids = list(input_ids)

    if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], (list, tuple, np.ndarray)):
        input_ids = input_ids[0]

    if not isinstance(input_ids, list):
        input_ids = list(input_ids)

    return [int(token) for token in input_ids]


def _tokenize_text(tokenizer: Any, text: str) -> list[int]:
    return _extract_input_ids(tokenizer(text))


def load_info_lines(info_file: str | Path) -> list[str]:
    info_path = resolve_local_path(info_file)
    with info_path.open("r", encoding="utf-8") as handle:
        return handle.readlines()


def build_semantic_prefix_allowed_token_map(
    info_lines: Sequence[str],
    *,
    tokenizer: Any,
    eos_token_id: int,
    base_model_hint: str,
) -> dict[str, list[int]]:
    semantic_ids = [line.split("\t")[0].strip() + "\n" for line in info_lines if line.strip()]
    prompt_texts = [f"### Response:\n{semantic_id}" for semantic_id in semantic_ids]

    is_llama = "llama" in base_model_hint.lower()
    tokenized_candidates: list[list[int]] = []
    for text in prompt_texts:
        token_ids = _tokenize_text(tokenizer, text)
        if is_llama:
            token_ids = token_ids[1:]
        tokenized_candidates.append(token_ids)

    return build_prefix_allowed_token_map(
        tokenized_candidates=tokenized_candidates,
        eos_token_id=int(eos_token_id),
        base_model=base_model_hint,
    )


def _parse_literal_list(raw_value: str) -> list[str]:
    parsed_value = ast.literal_eval(raw_value)
    if not isinstance(parsed_value, list):
        raise RuntimeError(f"Expected a list literal but got: {type(parsed_value)!r}")
    return [str(value) for value in parsed_value]


def build_eval_history_record(row: Mapping[str, str]) -> dict[str, Any]:
    history_item_sid = _parse_literal_list(str(row.get("history_item_sid", "[]")))
    history = ", ".join(history_item_sid)

    target_item_sid = str(row.get("item_sid", ""))
    last_history_item_sid = history_item_sid[-1] if history_item_sid else None

    return {
        "input": (
            "Can you predict the next possible item the user may expect, "
            "given the following chronological interaction history: "
            f"{history}"
        ),
        "output": target_item_sid + "\n",
        "dedup": target_item_sid == last_history_item_sid,
    }


def build_user_prompt(data_point: Mapping[str, str]) -> str:
    return USER_PROMPT_TEMPLATE.format(
        user_input=data_point["input"],
        response=data_point["output"],
    )


def build_eval_prompt_encoding(
    row: Mapping[str, str],
    *,
    tokenizer_adapter: OfficialTokenizerAdapter,
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    tokens = tokenizer_adapter.encode(INSTRUCTION_TEMPLATE, bos=True, eos=False)

    history = build_eval_history_record(row)
    prompt_point = dict(history)
    prompt_point["output"] = ""

    prompt = build_user_prompt(prompt_point)
    tokens = tokens + tokenizer_adapter.encode(prompt, bos=False, eos=False)

    return {
        "input_ids": tokens,
        "attention_mask": [1] * len(tokens),
    }, history


def load_test_rows(test_csv: str | Path, *, limit: int | None) -> list[dict[str, str]]:
    csv_path = resolve_local_path(test_csv)
    rows: list[dict[str, str]] = []

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(dict(row))
            if limit is not None and limit >= 0 and len(rows) >= limit:
                break

    return rows


def build_eval_payload(
    rows: Sequence[Mapping[str, str]],
    *,
    tokenizer_adapter: OfficialTokenizerAdapter,
) -> tuple[list[dict[str, list[int]]], list[dict[str, Any]]]:
    encodings: list[dict[str, list[int]]] = []
    outputs: list[dict[str, Any]] = []

    for row in rows:
        encoding, history = build_eval_prompt_encoding(row, tokenizer_adapter=tokenizer_adapter)
        encodings.append(encoding)
        outputs.append(history)

    return encodings, outputs


def left_pad_encodings(
    encodings: Sequence[Mapping[str, Sequence[int]]],
    *,
    pad_token_id: int,
    pad_to_multiple_of: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    if not encodings:
        raise RuntimeError("Cannot pad an empty batch.")

    max_len = max(len(encoding["input_ids"]) for encoding in encodings)
    if pad_to_multiple_of is not None and pad_to_multiple_of > 1:
        max_len = int(np.ceil(max_len / pad_to_multiple_of) * pad_to_multiple_of)
    padded_input_ids: list[list[int]] = []
    padded_attention_mask: list[list[int]] = []

    for encoding in encodings:
        input_ids = [int(token) for token in encoding["input_ids"]]
        pad_length = max_len - len(input_ids)
        padded_input_ids.append([int(pad_token_id)] * pad_length + input_ids)
        padded_attention_mask.append([0] * pad_length + [1] * len(input_ids))

    return (
        np.asarray(padded_input_ids, dtype=np.int32),
        np.asarray(padded_attention_mask, dtype=np.int32),
        max_len,
    )


def postprocess_and_group_outputs(decoded: Sequence[str], *, num_beams: int) -> list[list[str]]:
    normalized = [text.split("Response:\n")[-1].strip() for text in decoded]
    if len(normalized) % num_beams != 0:
        raise RuntimeError(
            "Decoded output count is not divisible by `num_beams`: "
            f"len={len(normalized)} num_beams={num_beams}"
        )

    return [
        normalized[index * num_beams : (index + 1) * num_beams]
        for index in range(len(normalized) // num_beams)
    ]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _ensure_generation_config_attrs(generation_config: Any) -> None:
    defaults = {
        "forced_decoder_ids": None,
        "suppress_tokens": None,
        "begin_suppress_tokens": None,
        "sequence_bias": None,
    }
    for name, value in defaults.items():
        if not hasattr(generation_config, name):
            setattr(generation_config, name, value)


def run_official_eval_parity(
    *,
    checkpoint_dir: str | Path,
    info_file: str | Path,
    test_csv: str | Path,
    result_json: str | Path,
    category: str,
    batch_size: int,
    num_beams: int,
    max_new_tokens: int,
    length_penalty: float,
    seed: int,
    limit: int | None,
    dry_run: bool,
    sharding_axis_dims: str | Sequence[int] = (1, 1, 1, -1, 1),
) -> OfficialEvalParityResult:
    resolved_checkpoint_dir = resolve_local_path(checkpoint_dir)
    resolved_info_file = resolve_local_path(info_file)
    resolved_test_csv = resolve_local_path(test_csv)
    resolved_result_json = resolve_local_path(result_json)
    category_text = resolve_category_text(category)

    parsed_sharding_axis_dims = parse_sharding_axis_dims(sharding_axis_dims)
    if batch_size <= 0:
        raise RuntimeError("`batch_size` must be positive.")
    if num_beams <= 0:
        raise RuntimeError("`num_beams` must be positive.")
    if max_new_tokens <= 0:
        raise RuntimeError("`max_new_tokens` must be positive.")

    warnings: list[str] = []

    easydel_available = _module_available("easydel")
    jax_available = _module_available("jax")
    transformers_available = _module_available("transformers")

    preview_sample_count = 0
    if resolved_test_csv.exists():
        try:
            preview_sample_count = len(load_test_rows(resolved_test_csv, limit=limit))
        except Exception as exc:
            warnings.append(f"failed_preview_test_csv={type(exc).__name__}:{exc}")
    else:
        warnings.append(f"test_csv_missing={resolved_test_csv}")

    if not resolved_info_file.exists():
        warnings.append(f"info_file_missing={resolved_info_file}")
    if not resolved_checkpoint_dir.exists():
        warnings.append(f"checkpoint_dir_missing={resolved_checkpoint_dir}")

    if dry_run:
        return OfficialEvalParityResult(
            dry_run=True,
            checkpoint_dir=resolved_checkpoint_dir,
            info_file=resolved_info_file,
            test_csv=resolved_test_csv,
            result_json=resolved_result_json,
            category=category,
            category_text=category_text,
            sample_count=preview_sample_count,
            written_count=0,
            batch_size=batch_size,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            length_penalty=length_penalty,
            seed=seed,
            limit=limit,
            sharding_axis_dims=parsed_sharding_axis_dims,
            easydel_available=easydel_available,
            jax_available=jax_available,
            transformers_available=transformers_available,
            warnings=tuple(warnings),
        )

    if not resolved_checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {resolved_checkpoint_dir}")
    if not resolved_info_file.is_file():
        raise FileNotFoundError(f"Info file does not exist: {resolved_info_file}")
    if not resolved_test_csv.is_file():
        raise FileNotFoundError(f"Test CSV does not exist: {resolved_test_csv}")

    if not easydel_available or not jax_available or not transformers_available:
        missing: list[str] = []
        if not easydel_available:
            missing.append("easydel")
        if not jax_available:
            missing.append("jax")
        if not transformers_available:
            missing.append("transformers")
        raise RuntimeError(
            "`eval-official-parity` requires runtime dependencies in non-dry-run mode. "
            f"Missing: {', '.join(missing)}"
        )

    _set_seed(seed)

    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec
    from transformers import AutoTokenizer, GenerationConfig

    try:
        from easydel.modules.auto.auto_modeling import AutoEasyDeLModelForCausalLM
    except Exception:
        from easydel import AutoEasyDeLModelForCausalLM

    from easydel.inference.logits_process import LogitsProcessorList

    tokenizer = AutoTokenizer.from_pretrained(str(resolved_checkpoint_dir))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model_load_kwargs = {
        "from_torch": True,
        "auto_shard_model": False,
        "dtype": jnp.bfloat16,
        "param_dtype": jnp.bfloat16,
        "quantize_tensors": False,
        "verbose": False,
    }

    effective_sharding_axis_dims = parsed_sharding_axis_dims
    try:
        model = AutoEasyDeLModelForCausalLM.from_pretrained(
            str(resolved_checkpoint_dir),
            sharding_axis_dims=effective_sharding_axis_dims,
            **model_load_kwargs,
        )
    except ValueError as exc:
        if not _should_retry_mesh_with_4d_axis_dims(exc, parsed_sharding_axis_dims):
            raise

        effective_sharding_axis_dims = parsed_sharding_axis_dims[:4]
        warnings.append(
            "mesh_axis_dims_autofix=retry_with_4d_axis_dims"
            f"({parsed_sharding_axis_dims}->{effective_sharding_axis_dims})"
        )
        model = AutoEasyDeLModelForCausalLM.from_pretrained(
            str(resolved_checkpoint_dir),
            sharding_axis_dims=effective_sharding_axis_dims,
            **model_load_kwargs,
        )
    model.eval()

    _ensure_generation_config_attrs(model.generation_config)

    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    if tokenizer.bos_token_id is not None:
        model.config.bos_token_id = tokenizer.bos_token_id

    base_model_hint = f"{resolved_checkpoint_dir.name}-{getattr(model.config, 'model_type', '')}"

    info_lines = load_info_lines(resolved_info_file)
    prefix_token_map = build_semantic_prefix_allowed_token_map(
        info_lines,
        tokenizer=tokenizer,
        eos_token_id=int(tokenizer.eos_token_id),
        base_model_hint=base_model_hint,
    )
    prefix_allowed_tokens_fn = build_prefix_allowed_tokens_fn(prefix_token_map)

    test_rows = load_test_rows(resolved_test_csv, limit=limit)
    tokenizer_adapter = OfficialTokenizerAdapter(tokenizer)
    encodings, output_rows = build_eval_payload(test_rows, tokenizer_adapter=tokenizer_adapter)

    grouped_predictions: list[list[str]] = []
    batch_count = (len(encodings) + batch_size - 1) // batch_size

    resolved_axis_dims = _resolve_axis_dims_for_device_count(
        effective_sharding_axis_dims,
        device_count=int(jax.device_count()),
    )
    sequence_partition_factor = 1
    if len(resolved_axis_dims) >= 4:
        sequence_partition_factor = int(resolved_axis_dims[3])
    pad_to_multiple_of = sequence_partition_factor if sequence_partition_factor > 1 else None
    if pad_to_multiple_of is not None:
        warnings.append(f"input_pad_multiple_of={pad_to_multiple_of}")

    for batch_index in range(batch_count):
        batch_encodings = encodings[batch_index * batch_size : (batch_index + 1) * batch_size]
        padded_input_ids, padded_attention_mask, max_prompt_len = left_pad_encodings(
            batch_encodings,
            pad_token_id=int(tokenizer.pad_token_id),
            pad_to_multiple_of=pad_to_multiple_of,
        )

        generation_config = GenerationConfig(
            num_beams=int(num_beams),
            num_return_sequences=int(num_beams),
            length_penalty=float(length_penalty),
            pad_token_id=int(model.config.pad_token_id),
            eos_token_id=int(model.config.eos_token_id),
            max_new_tokens=int(max_new_tokens),
            top_k=None,
            top_p=None,
        )
        _ensure_generation_config_attrs(generation_config)

        constrained_processor = ConstrainedLogitsProcessor(
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=num_beams,
            base_model=base_model_hint,
            eos_token_id=int(model.config.eos_token_id),
        )
        logits_processor = LogitsProcessorList([constrained_processor])

        if not hasattr(model, "mesh"):
            raise RuntimeError("Loaded EasyDeL model does not expose `.mesh` required for generation.")

        replicated_batch_sharding = NamedSharding(model.mesh, PartitionSpec(None, None))
        input_ids_array = jax.device_put(jnp.asarray(padded_input_ids), replicated_batch_sharding)
        attention_mask_array = jax.device_put(
            jnp.asarray(padded_attention_mask),
            replicated_batch_sharding,
        )

        with model.mesh:
            generation_output = model.generate(
                input_ids=input_ids_array,
                attention_mask=attention_mask_array,
                generation_config=generation_config,
                logits_processor=logits_processor,
            )

        sequences = getattr(generation_output, "sequences", generation_output)
        completions = np.asarray(sequences)[:, max_prompt_len:]

        if "llama" in base_model_hint.lower():
            decoded = tokenizer.batch_decode(
                completions,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        else:
            decoded = tokenizer.batch_decode(completions, skip_special_tokens=True)

        grouped_predictions.extend(postprocess_and_group_outputs(decoded, num_beams=num_beams))

    if len(grouped_predictions) != len(output_rows):
        raise RuntimeError(
            "Prediction row count mismatch. "
            f"predictions={len(grouped_predictions)} outputs={len(output_rows)}"
        )

    for index, row in enumerate(output_rows):
        row["predict"] = grouped_predictions[index]
        if "dedup" in row:
            row.pop("dedup")

    resolved_result_json.parent.mkdir(parents=True, exist_ok=True)
    with resolved_result_json.open("w", encoding="utf-8") as handle:
        json.dump(output_rows, handle, indent=4, ensure_ascii=False)

    return OfficialEvalParityResult(
        dry_run=False,
        checkpoint_dir=resolved_checkpoint_dir,
        info_file=resolved_info_file,
        test_csv=resolved_test_csv,
        result_json=resolved_result_json,
        category=category,
        category_text=category_text,
        sample_count=len(encodings),
        written_count=len(output_rows),
        batch_size=batch_size,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        length_penalty=length_penalty,
        seed=seed,
        limit=limit,
        sharding_axis_dims=effective_sharding_axis_dims,
        easydel_available=easydel_available,
        jax_available=jax_available,
        transformers_available=transformers_available,
        warnings=tuple(warnings),
    )


__all__ = [
    "CATEGORY_TEXT_MAP",
    "INSTRUCTION_TEMPLATE",
    "OfficialEvalParityResult",
    "OfficialTokenizerAdapter",
    "build_eval_history_record",
    "build_eval_payload",
    "build_eval_prompt_encoding",
    "build_semantic_prefix_allowed_token_map",
    "left_pad_encodings",
    "parse_sharding_axis_dims",
    "postprocess_and_group_outputs",
    "run_official_eval_parity",
]
