from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import rank
from demo_ranker import load_demo_candidates
from redrob_ranker.features import (
    NUMERIC_FEATURES,
    career_narrative_features,
    description_frequency_band,
    extract_features,
    reasoning_for_row,
)
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

    def test_description_frequency_bands_are_stable(self) -> None:
        frequencies = [0, 25_000, 10_000, 1_800, 350, 60, 10]
        self.assertEqual(
            [description_frequency_band(value) for value in frequencies],
            [0, 0, 1, 2, 3, 4, 5],
        )

    def test_rare_expert_narrative_requires_semantic_evidence(self) -> None:
        relevant = "Owned a production ranking layer with offline evaluation metrics."
        irrelevant = "Managed quarterly accounting and statutory tax filings."
        expert = career_narrative_features(
            [{"description": relevant, "is_current": True}],
            {relevant: 5},
        )
        nonexpert = career_narrative_features(
            [{"description": irrelevant, "is_current": True}],
            {irrelevant: 5},
        )
        self.assertEqual(expert["rare_expert_narrative"], 1.0)
        self.assertEqual(nonexpert["rare_expert_narrative"], 0.0)

    def test_reasoning_does_not_overstate_weak_sample_candidate(self) -> None:
        reasoning = reasoning_for_row(
            {
                "current_title": "Cloud Engineer",
                "years_experience": 8.3,
                "reason_strongest_evidence": "Maintained cloud infrastructure.",
                "reason_concern": "limited explicit production ML evidence",
                "location": "Chandigarh",
                "recruiter_response_rate": 0.50,
                "notice_period_days": 120,
                "retrieval_ranking_depth": 0.05,
                "production_ml_depth": 0.10,
                "evaluation_experimentation_fit": 0.0,
                "product_company_fit": 0.30,
                "career_system_fit": 0.20,
            },
            rank=2,
        )
        self.assertNotIn("top-tier", reasoning.lower())
        self.assertIn("concern", reasoning.lower())

    def test_zero_duration_expert_anomalies_cover_non_jd_skills(self) -> None:
        features = extract_features(
            {
                "candidate_id": "CAND_9999999",
                "profile": {"years_of_experience": 2.0},
                "skills": [
                    {"name": "Accounting", "proficiency": "expert", "duration_months": 0},
                    {"name": "Brand Design", "proficiency": "expert", "duration_months": 1},
                ],
                "career_history": [],
                "redrob_signals": {},
            }
        )
        self.assertEqual(features["suspicious_expert_skill_count"], 2)

    def test_candidate_id_breaks_equal_score_ties(self) -> None:
        frame = pd.DataFrame(
            {
                "candidate_id": ["CAND_0000002", "CAND_0000001"],
                "final_score": [0.9, 0.9],
            }
        )
        ranked = rank.select_top_candidates(frame, top_n=2)
        self.assertEqual(
            ranked["candidate_id"].tolist(),
            ["CAND_0000001", "CAND_0000002"],
        )

    def test_demo_rejects_duplicate_candidate_ids(self) -> None:
        candidate = {
            "candidate_id": "CAND_0000001",
            "profile": {},
            "career_history": [],
            "education": [],
            "skills": [],
            "redrob_signals": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicates.json"
            path.write_text(json.dumps([candidate, candidate]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unique"):
                load_demo_candidates(path)

    def test_demo_rejects_invalid_candidate_id(self) -> None:
        candidate = {
            "candidate_id": "candidate-1",
            "profile": {},
            "career_history": [],
            "education": [],
            "skills": [],
            "redrob_signals": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps([candidate]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "seven digits"):
                load_demo_candidates(path)

    def test_demo_rejects_oversized_upload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oversized.json"
            path.write_bytes(b" " * 5_000_001)
            with self.assertRaisesRegex(ValueError, "5 MB"):
                load_demo_candidates(path)

    def test_reasoning_uses_supplied_facts_and_rank_tone(self) -> None:
        reasoning = reasoning_for_row(
            {
                "current_title": "Search Engineer",
                "years_experience": 7.2,
                "reason_strongest_evidence": "Built hybrid retrieval with NDCG evaluation.",
                "reason_concern": "60-day notice period",
                "location": "Pune, Maharashtra",
                "recruiter_response_rate": 0.82,
                "notice_period_days": 60,
                "retrieval_ranking_depth": 0.95,
                "production_ml_depth": 0.80,
                "evaluation_experimentation_fit": 0.90,
                "product_company_fit": 0.70,
            },
            rank=92,
        )
        self.assertIn("top-100", reasoning.lower())
        self.assertIn("Search Engineer", reasoning)
        self.assertIn("7.2 yrs", reasoning)
        self.assertIn("hybrid retrieval", reasoning)
        self.assertIn("Pune, Maharashtra", reasoning)
        self.assertIn("60-day notice", reasoning)


if __name__ == "__main__":
    unittest.main()
