from __future__ import annotations

import math
import unittest

from minionerec_jax.eval_metrics import compute_offline_metrics, format_metric_lines


class EvalMetricsTest(unittest.TestCase):
    def test_perfect_hit_at_rank0(self) -> None:
        rows = [{"predict": ["item-a", "item-b", "item-c"], "output": "item-a"}]

        result = compute_offline_metrics(rows)

        self.assertEqual(result.n_beam, 3)
        self.assertEqual(result.valid_topk, (1, 3))
        self.assertAlmostEqual(result.hr[1], 1.0)
        self.assertAlmostEqual(result.hr[3], 1.0)
        self.assertAlmostEqual(result.ndcg[1], 1.0)
        self.assertAlmostEqual(result.ndcg[3], 1.0)

    def test_hit_at_later_rank(self) -> None:
        rows = [{"predict": ["item-a", "item-b", "item-c", "item-d"], "output": "item-c"}]

        result = compute_offline_metrics(rows, topk_list=(1, 3, 5))

        self.assertEqual(result.valid_topk, (1, 3))
        self.assertAlmostEqual(result.hr[1], 0.0)
        self.assertAlmostEqual(result.hr[3], 1.0)
        self.assertAlmostEqual(result.ndcg[1], 0.0)
        self.assertAlmostEqual(result.ndcg[3], math.log(2) / math.log(4))

    def test_no_hit_case(self) -> None:
        rows = [{"predict": ["item-a", "item-b", "item-c"], "output": "item-z"}]

        result = compute_offline_metrics(rows, topk_list=(1, 3, 5))

        self.assertEqual(result.valid_topk, (1, 3))
        self.assertAlmostEqual(result.hr[1], 0.0)
        self.assertAlmostEqual(result.hr[3], 0.0)
        self.assertAlmostEqual(result.ndcg[1], 0.0)
        self.assertAlmostEqual(result.ndcg[3], 0.0)

    def test_mixed_output_types(self) -> None:
        rows = [
            {"predict": ["item-a", "item-b", "item-c"], "output": "item-a"},
            {"predict": ["item-a", "item-b", "item-c"], "output": ["item-a"]},
        ]

        result = compute_offline_metrics(rows)

        self.assertEqual(result.valid_topk, (1, 3))
        self.assertAlmostEqual(result.hr[1], 1.0)
        self.assertAlmostEqual(result.hr[3], 1.0)
        self.assertAlmostEqual(result.ndcg[1], 1.0)
        self.assertAlmostEqual(result.ndcg[3], 1.0)

    def test_n_beam_truncates_topk(self) -> None:
        rows = [{"predict": ["item-a", "item-b"], "output": "item-b"}]

        result = compute_offline_metrics(rows)
        metric_lines = format_metric_lines(result)

        self.assertEqual(result.n_beam, 2)
        self.assertEqual(result.valid_topk, (1,))
        self.assertAlmostEqual(result.hr[1], 0.0)
        self.assertAlmostEqual(result.ndcg[1], 0.0)
        self.assertEqual(metric_lines, ["hr@1=0.000000000000", "ndcg@1=0.000000000000"])


if __name__ == "__main__":
    unittest.main()
