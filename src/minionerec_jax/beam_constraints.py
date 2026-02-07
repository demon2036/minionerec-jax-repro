"""Constrained-beam logits masking utilities with MiniOneRec parity semantics."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any
import warnings

import numpy as np

try:
    import jax.numpy as jnp
except Exception:
    jnp = None


PrefixAllowedTokensFn = Callable[[int, Sequence[int]], Sequence[int]]


def get_prefix_index(base_model: str | None) -> int:
    model_name = (base_model or "").lower()
    if "gpt2" in model_name:
        return 4
    return 3


def hash_tokens(tokens: Sequence[int]) -> str:
    return "-".join(str(int(token)) for token in tokens)


def build_prefix_allowed_token_map(
    tokenized_candidates: Sequence[Sequence[int]],
    *,
    eos_token_id: int,
    base_model: str | None,
) -> dict[str, list[int]]:
    prefix_index = get_prefix_index(base_model)
    token_map: dict[str, set[int]] = {}
    for candidate_tokens in tokenized_candidates:
        candidate = [int(token) for token in candidate_tokens]
        candidate.append(int(eos_token_id))
        for i in range(prefix_index, len(candidate)):
            if i == prefix_index:
                prefix_tokens = candidate[:i]
            else:
                prefix_tokens = candidate[prefix_index:i]
            key = hash_tokens(prefix_tokens)
            if key not in token_map:
                token_map[key] = set()
            token_map[key].add(int(candidate[i]))
    return {key: sorted(values) for key, values in token_map.items()}


def build_prefix_allowed_tokens_fn(
    prefix_allowed_token_map: Mapping[str, Sequence[int]],
) -> PrefixAllowedTokensFn:
    normalized_map = {
        str(key): tuple(int(token) for token in values)
        for key, values in prefix_allowed_token_map.items()
    }

    def prefix_allowed_tokens_fn(batch_id: int, input_ids: Sequence[int]) -> list[int]:
        _ = batch_id
        return list(normalized_map.get(hash_tokens(input_ids), ()))

    return prefix_allowed_tokens_fn


def _is_jax_array(value: Any) -> bool:
    module_name = type(value).__module__
    return module_name.startswith("jax") or module_name.startswith("jaxlib")


def _resolve_backend(scores: Any, use_jax: bool | None) -> str:
    if use_jax is True:
        if jnp is None:
            raise RuntimeError("`use_jax=True` requires `jax` to be installed.")
        return "jax"
    if use_jax is False:
        return "numpy"
    if jnp is not None and _is_jax_array(scores):
        return "jax"
    return "numpy"


def _to_backend_array(value: Any, backend: str) -> Any:
    if backend == "jax":
        return jnp.asarray(value)
    return np.asarray(value)


def _log_softmax(scores: Any, backend: str) -> Any:
    xp = jnp if backend == "jax" else np
    shifted_scores = scores - xp.max(scores, axis=-1, keepdims=True)
    return shifted_scores - xp.log(xp.sum(xp.exp(shifted_scores), axis=-1, keepdims=True))


def _set_mask_tokens(mask: Any, *, row_index: int, token_ids: Sequence[int], backend: str) -> Any:
    if backend == "jax":
        return mask.at[row_index, jnp.asarray(list(token_ids), dtype=jnp.int32)].set(0.0)

    np_token_ids = np.asarray(list(token_ids), dtype=np.int64)
    mask[row_index, np_token_ids] = 0.0
    return mask


class ConstrainedLogitsProcessor:
    """Stateful logits processor mirroring official MiniOneRec constrained masking."""

    def __init__(
        self,
        prefix_allowed_tokens_fn: PrefixAllowedTokensFn,
        num_beams: int,
        *,
        base_model: str | None = None,
        eos_token_id: int | None = None,
    ) -> None:
        if num_beams <= 0:
            raise ValueError("`num_beams` must be positive.")
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self._num_beams = int(num_beams)
        self.count = 0
        self.base_model = base_model
        self.eos_token_id = None if eos_token_id is None else int(eos_token_id)
        self.prefix_index = get_prefix_index(base_model)

    def reset(self) -> None:
        self.count = 0

    def __call__(
        self,
        input_ids: Sequence[Sequence[int]] | np.ndarray,
        scores: Any,
        cur_len: int | Any | None = None,
    ) -> Any:
        _ = cur_len
        backend = _resolve_backend(scores, None)
        scores_array = _to_backend_array(scores, backend)
        if scores_array.ndim != 2:
            raise ValueError("`scores` must have shape (batch_size*num_beams, vocab_size).")

        input_ids_array = np.asarray(input_ids, dtype=np.int64)
        if input_ids_array.ndim != 2:
            raise ValueError("`input_ids` must have shape (batch_size*num_beams, sequence_length).")
        if input_ids_array.shape[0] != scores_array.shape[0]:
            raise ValueError(
                "`input_ids` and `scores` must agree on the first dimension "
                "(batch_size*num_beams)."
            )
        if input_ids_array.shape[0] % self._num_beams != 0:
            raise ValueError("`input_ids.shape[0]` must be divisible by `num_beams`.")

        scores_log = _log_softmax(scores_array, backend)
        if backend == "jax":
            mask = jnp.full_like(scores_log, -jnp.inf)
        else:
            mask = np.full_like(scores_log, -np.inf)

        for batch_id, beam_sent in enumerate(
            input_ids_array.reshape(-1, self._num_beams, input_ids_array.shape[-1])
        ):
            for beam_id, sent in enumerate(beam_sent):
                if self.count == 0:
                    hash_key = sent[-self.prefix_index:]
                else:
                    hash_key = sent[-self.count:]
                prefix_allowed_tokens = self._prefix_allowed_tokens_fn(batch_id, hash_key.tolist())

                row_index = batch_id * self._num_beams + beam_id
                if len(prefix_allowed_tokens) == 0:
                    warnings.warn(
                        f"No valid tokens found for hash_key {hash_key.tolist()} at step {self.count}. "
                        "This indicates the model generated an unexpected token. "
                    )
                    if self.eos_token_id is not None:
                        mask = _set_mask_tokens(
                            mask,
                            row_index=row_index,
                            token_ids=[self.eos_token_id],
                            backend=backend,
                        )
                    continue

                mask = _set_mask_tokens(
                    mask,
                    row_index=row_index,
                    token_ids=prefix_allowed_tokens,
                    backend=backend,
                )

        self.count += 1
        return scores_log + mask
