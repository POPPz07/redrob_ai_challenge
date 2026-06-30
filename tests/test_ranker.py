from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import rank
from redrob_ranker.features import NUMERIC_FEATURES
from redrob_ranker.late_interaction import score_late_interaction


class RankerUnitTests(unittest.TestCase):
    def test_calibrated_scores_are_strictly_descending(self) -> None:
        scores = rank.calibrated_output_scores(np.array([1.0, 1.0, 0.8, 0.2]))
        self.assertTrue(np.all(scores[:-1] > scores[1:]))

    def test_missing_model_uses_formula_fallback(self) -> None:
        row = {column: 0.5 for column in NUMERIC_FEATURES}
        row["weighted_formula_score"] = 0.75
        with tempfile.TemporaryDirectory() as directory:
            predicted = rank.predict_scores(pd.DataFrame([row]), Path(directory))
        self.assertEqual(predicted.shape, (1,))
        self.assertTrue(np.isfinite(predicted[0]))

    def test_lexical_colbert_fallback_is_available(self) -> None:
        result = score_late_interaction(
            "Shipped a production vector retrieval and ranking system with NDCG evaluation.",
            "Python search engineer",
        )
        self.assertEqual(result["colbert_available"], 0.0)
        self.assertGreater(result["colbert_maxsim_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
