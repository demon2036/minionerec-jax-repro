from __future__ import annotations

import unittest
import warnings

import numpy as np

from minionerec_jax.beam_constraints import (
    ConstrainedLogitsProcessor,
    build_prefix_allowed_token_map,
    build_prefix_allowed_tokens_fn,
    get_prefix_index,
    hash_tokens,
)


def _log_softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


class BeamConstraintsTest(unittest.TestCase):
    def test_valid_prefix_masking(self) -> None:
        eos_token_id = 9
        prefix_map = build_prefix_allowed_token_map(
            tokenized_candidates=((1, 2, 3, 4), (1, 2, 3, 5)),
            eos_token_id=eos_token_id,
            base_model="llama-3",
        )
        self.assertEqual(prefix_map["1-2-3"], [4, 5])

        processor = ConstrainedLogitsProcessor(
            prefix_allowed_tokens_fn=build_prefix_allowed_tokens_fn(prefix_map),
            num_beams=2,
            base_model="llama-3",
            eos_token_id=eos_token_id,
        )

        input_ids = np.asarray(
            [
                [8, 1, 2, 3],
                [7, 1, 2, 3],
            ],
            dtype=np.int64,
        )
        scores = np.asarray(
            [
                [0.1, 0.2, 0.3, 0.4, 1.7, 1.2, 0.1, 0.0, 0.3, 0.5],
                [0.3, 0.4, 0.1, 0.2, 1.4, 1.1, 0.0, 0.1, 0.3, 0.6],
            ],
            dtype=np.float64,
        )

        output = np.asarray(processor(input_ids, scores), dtype=np.float64)
        expected = _log_softmax(scores)

        self.assertEqual(processor.count, 1)
        for row_index in range(output.shape[0]):
            finite_tokens = np.where(np.isfinite(output[row_index]))[0].tolist()
            self.assertEqual(finite_tokens, [4, 5])
            np.testing.assert_allclose(output[row_index, [4, 5]], expected[row_index, [4, 5]])
            disallowed = [token for token in range(output.shape[-1]) if token not in (4, 5)]
            self.assertTrue(np.all(np.isneginf(output[row_index, disallowed])))

    def test_empty_allowed_set_falls_back_to_eos(self) -> None:
        eos_token_id = 7
        processor = ConstrainedLogitsProcessor(
            prefix_allowed_tokens_fn=lambda _batch_id, _tokens: [],
            num_beams=1,
            base_model="llama",
            eos_token_id=eos_token_id,
        )

        input_ids = np.asarray([[1, 2, 3]], dtype=np.int64)
        scores = np.asarray([[0.1, 0.2, 0.3, 0.4, 0.7, 0.6, 0.5, 0.8, 0.0, -0.1]], dtype=np.float64)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            output = np.asarray(processor(input_ids, scores), dtype=np.float64)

        expected = _log_softmax(scores)
        finite_tokens = np.where(np.isfinite(output[0]))[0].tolist()
        self.assertEqual(finite_tokens, [eos_token_id])
        self.assertAlmostEqual(output[0, eos_token_id], expected[0, eos_token_id])
        self.assertEqual(processor.count, 1)
        self.assertGreaterEqual(len(caught), 1)
        self.assertIn("No valid tokens found", str(caught[0].message))

    def test_count_updates_and_hash_window_progression(self) -> None:
        calls: list[tuple[int, list[int]]] = []

        def prefix_allowed_tokens_fn(batch_id: int, token_window: list[int]) -> list[int]:
            calls.append((batch_id, list(token_window)))
            token_map = {
                "1-2-3": [2],
                "4-5-6": [3],
                "7": [4],
                "8": [5],
            }
            return token_map.get(hash_tokens(token_window), [])

        processor = ConstrainedLogitsProcessor(
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=2,
            base_model="llama",
            eos_token_id=9,
        )

        input_ids_step0 = np.asarray(
            [
                [9, 1, 2, 3],
                [9, 4, 5, 6],
            ],
            dtype=np.int64,
        )
        scores_step0 = np.asarray(
            [
                [0.1, 0.2, 1.8, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2],
                [0.2, 0.1, 0.4, 1.9, 0.3, 0.2, 0.1, 0.0, -0.2, -0.3],
            ],
            dtype=np.float64,
        )
        output_step0 = np.asarray(processor(input_ids_step0, scores_step0), dtype=np.float64)
        self.assertEqual(processor.count, 1)
        self.assertEqual(np.where(np.isfinite(output_step0[0]))[0].tolist(), [2])
        self.assertEqual(np.where(np.isfinite(output_step0[1]))[0].tolist(), [3])

        input_ids_step1 = np.asarray(
            [
                [9, 1, 2, 3, 7],
                [9, 4, 5, 6, 8],
            ],
            dtype=np.int64,
        )
        scores_step1 = np.asarray(
            [
                [0.1, 0.2, 0.3, 0.4, 1.9, 0.1, 0.0, 0.2, -0.1, -0.2],
                [0.2, 0.1, 0.3, 0.4, 0.1, 1.9, 0.0, 0.2, -0.1, -0.3],
            ],
            dtype=np.float64,
        )
        output_step1 = np.asarray(processor(input_ids_step1, scores_step1), dtype=np.float64)
        self.assertEqual(processor.count, 2)
        self.assertEqual(np.where(np.isfinite(output_step1[0]))[0].tolist(), [4])
        self.assertEqual(np.where(np.isfinite(output_step1[1]))[0].tolist(), [5])

        self.assertEqual(
            calls,
            [
                (0, [1, 2, 3]),
                (0, [4, 5, 6]),
                (0, [7]),
                (0, [8]),
            ],
        )

    def test_cur_len_slices_effective_sequence_for_easydel_padded_state(self) -> None:
        calls: list[list[int]] = []

        def prefix_allowed_tokens_fn(_batch_id: int, token_window: list[int]) -> list[int]:
            calls.append(list(token_window))
            token_map = {
                "1-2-3": [4],
                "4-5-6": [5],
                "7": [8],
                "9": [6],
            }
            return token_map.get(hash_tokens(token_window), [])

        processor = ConstrainedLogitsProcessor(
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=2,
            base_model="llama",
            eos_token_id=0,
        )

        eos_pad = 151645
        input_ids_step0 = np.asarray(
            [
                [eos_pad, eos_pad, 1, 2, 3, eos_pad, eos_pad, eos_pad],
                [eos_pad, eos_pad, 4, 5, 6, eos_pad, eos_pad, eos_pad],
            ],
            dtype=np.int64,
        )
        scores_step0 = np.asarray(
            [
                [0.1, 0.2, 0.3, 0.4, 1.8, 0.1, 0.0, -0.1, -0.2, -0.3],
                [0.1, 0.2, 0.3, 0.4, 0.1, 1.8, 0.0, -0.1, -0.2, -0.3],
            ],
            dtype=np.float64,
        )
        output_step0 = np.asarray(processor(input_ids_step0, scores_step0, cur_len=5), dtype=np.float64)
        self.assertEqual(np.where(np.isfinite(output_step0[0]))[0].tolist(), [4])
        self.assertEqual(np.where(np.isfinite(output_step0[1]))[0].tolist(), [5])

        input_ids_step1 = np.asarray(
            [
                [eos_pad, eos_pad, 1, 2, 3, 7, eos_pad, eos_pad],
                [eos_pad, eos_pad, 4, 5, 6, 9, eos_pad, eos_pad],
            ],
            dtype=np.int64,
        )
        scores_step1 = np.asarray(
            [
                [0.1, 0.2, 0.3, 0.4, 0.0, 0.1, 0.2, 0.3, 1.9, 0.0],
                [0.1, 0.2, 0.3, 0.4, 0.0, 0.1, 1.9, 0.3, 0.2, 0.0],
            ],
            dtype=np.float64,
        )
        output_step1 = np.asarray(processor(input_ids_step1, scores_step1, cur_len=6), dtype=np.float64)
        self.assertEqual(np.where(np.isfinite(output_step1[0]))[0].tolist(), [8])
        self.assertEqual(np.where(np.isfinite(output_step1[1]))[0].tolist(), [6])

        self.assertEqual(calls, [[1, 2, 3], [4, 5, 6], [7], [9]])

    def test_gpt2_prefix_index_branch(self) -> None:
        gpt2_calls: list[list[int]] = []

        def gpt2_fn(_batch_id: int, token_window: list[int]) -> list[int]:
            gpt2_calls.append(list(token_window))
            return [0]

        gpt2_processor = ConstrainedLogitsProcessor(
            prefix_allowed_tokens_fn=gpt2_fn,
            num_beams=1,
            base_model="gpt2-medium",
            eos_token_id=9,
        )
        gpt2_processor(
            np.asarray([[1, 2, 3, 4, 5]], dtype=np.int64),
            np.asarray([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], dtype=np.float64),
        )

        non_gpt2_calls: list[list[int]] = []

        def non_gpt2_fn(_batch_id: int, token_window: list[int]) -> list[int]:
            non_gpt2_calls.append(list(token_window))
            return [0]

        non_gpt2_processor = ConstrainedLogitsProcessor(
            prefix_allowed_tokens_fn=non_gpt2_fn,
            num_beams=1,
            base_model="llama-3",
            eos_token_id=9,
        )
        non_gpt2_processor(
            np.asarray([[1, 2, 3, 4, 5]], dtype=np.int64),
            np.asarray([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], dtype=np.float64),
        )

        self.assertEqual(get_prefix_index("gpt2-medium"), 4)
        self.assertEqual(get_prefix_index("llama-3"), 3)
        self.assertEqual(gpt2_processor.prefix_index, 4)
        self.assertEqual(non_gpt2_processor.prefix_index, 3)
        self.assertEqual(gpt2_calls, [[2, 3, 4, 5]])
        self.assertEqual(non_gpt2_calls, [[3, 4, 5]])


if __name__ == "__main__":
    unittest.main()
