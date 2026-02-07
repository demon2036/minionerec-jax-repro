from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from minionerec_jax.official_eval_parity import (
    INSTRUCTION_TEMPLATE,
    OfficialTokenizerAdapter,
    build_eval_prompt_encoding,
    build_semantic_prefix_allowed_token_map,
    parse_sharding_axis_dims,
    postprocess_and_group_outputs,
    run_official_eval_parity,
)


class _FakeTokenizer:
    bos_token_id = 101
    eos_token_id = 102
    pad_token_id = 0
    eos_token = "</s>"
    pad_token = "<pad>"

    def __init__(self, mapping: dict[str, list[int]]):
        self._mapping = mapping

    def encode(self, text: str) -> list[int]:
        if text not in self._mapping:
            raise KeyError(f"Missing fake tokenization fixture for: {text!r}")
        return list(self._mapping[text])

    def __call__(self, text: str) -> dict[str, list[int]]:
        return {"input_ids": self.encode(text)}


class OfficialEvalParityTest(unittest.TestCase):
    def test_prompt_encoding_matches_official_bos_eos_rules(self) -> None:
        row = {
            "history_item_sid": "['<a_1><b_2><c_3>', '<a_1><b_2><c_4>']",
            "item_sid": "<a_9><b_9><c_9>",
        }
        prompt = (
            "### User Input: \n"
            "Can you predict the next possible item the user may expect, given the following "
            "chronological interaction history: <a_1><b_2><c_3>, <a_1><b_2><c_4>\n\n"
            "### Response:\n"
        )
        tokenizer = _FakeTokenizer(
            {
                INSTRUCTION_TEMPLATE: [101, 11, 12, 102],
                prompt: [101, 21, 22, 102],
            }
        )
        adapter = OfficialTokenizerAdapter(tokenizer)

        encoding, history = build_eval_prompt_encoding(row, tokenizer_adapter=adapter)

        self.assertEqual(encoding["input_ids"], [101, 11, 12, 21, 22])
        self.assertEqual(encoding["attention_mask"], [1, 1, 1, 1, 1])
        self.assertEqual(history["output"], "<a_9><b_9><c_9>\n")
        self.assertFalse(history["dedup"])

    def test_semantic_prefix_map_uses_llama_token_drop_branch(self) -> None:
        info_lines = [
            "<sid-a>\tItem A\t0\n",
            "<sid-b>\tItem B\t1\n",
        ]
        tokenizer = _FakeTokenizer(
            {
                "### Response:\n<sid-a>\n": [500, 10, 11, 12, 13],
                "### Response:\n<sid-b>\n": [500, 10, 11, 12, 14],
            }
        )

        token_map = build_semantic_prefix_allowed_token_map(
            info_lines,
            tokenizer=tokenizer,
            eos_token_id=99,
            base_model_hint="llama-3",
        )

        self.assertEqual(token_map["10-11-12"], [13, 14])

    def test_decode_postprocess_and_group(self) -> None:
        grouped = postprocess_and_group_outputs(
            [
                "noise Response:\n<a_1>",
                "### Response:\n<a_2>\n",
                "anything Response:\n<a_3>",
                "Response:\n<a_4>",
            ],
            num_beams=2,
        )

        self.assertEqual(grouped, [["<a_1>", "<a_2>"], ["<a_3>", "<a_4>"]])

    def test_parse_sharding_dims_and_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint_dir = root / "ckpt"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            info_file = root / "info.txt"
            info_file.write_text("<sid-a>\tItem A\t0\n", encoding="utf-8")

            test_csv = root / "test.csv"
            test_csv.write_text(
                "user_id,history_item_sid,item_sid\n"
                "u1,\"['<sid-a>']\",<sid-b>\n",
                encoding="utf-8",
            )

            result_json = root / "out.json"

            result = run_official_eval_parity(
                checkpoint_dir=checkpoint_dir,
                info_file=info_file,
                test_csv=test_csv,
                result_json=result_json,
                category="Industrial_and_Scientific",
                batch_size=2,
                num_beams=4,
                max_new_tokens=8,
                length_penalty=0.0,
                seed=42,
                limit=None,
                dry_run=True,
                sharding_axis_dims="1,1,1,-1,1",
            )

        self.assertEqual(parse_sharding_axis_dims("1,1,1,-1,1"), (1, 1, 1, -1, 1))
        self.assertTrue(result.dry_run)
        self.assertEqual(result.sample_count, 1)
        self.assertEqual(result.written_count, 0)
        self.assertEqual(result.category_text, "industrial and scientific items")


if __name__ == "__main__":
    unittest.main()
