from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from redrob_ranker.features import extract_features, reasoning_for_row, stable_minmax
from redrob_ranker.late_interaction import score_late_interaction


def load_demo_candidates(path: str | Path) -> list[dict]:
    source = Path(path)
    text = source.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError("Candidate file is empty.")
    if text.startswith("["):
        candidates = json.loads(text)
    else:
        candidates = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Expected a JSON list or JSONL candidate file.")
    if len(candidates) > 100:
        raise ValueError("The hosted demo accepts at most 100 candidates.")
    return candidates


def rank_demo_file(candidate_path: str | Path, output_path: str | Path) -> Path:
    candidates = load_demo_candidates(candidate_path)
    rows = [extract_features(candidate) for candidate in candidates]
    for row in rows:
        late = score_late_interaction(row["evidence_text"], row["career_text"])
        score = float(row["weighted_formula_score"]) + 0.055 * late["colbert_maxsim_score"]
        score += 0.025 * float(row["production_ml_depth"])
        score += 0.015 * float(row["product_company_fit"])
        score += 0.010 * float(row["experience_fit"])
        if row["honeypot_risk"] >= 0.30 and row["consistency_score"] < 0.65:
            score *= 0.05
        row["demo_score"] = score

    ranked = sorted(rows, key=lambda row: (-row["demo_score"], row["candidate_id"]))
    raw_scores = np.array([row["demo_score"] for row in ranked], dtype=np.float64)
    normalized = np.asarray(stable_minmax(raw_scores.tolist()), dtype=np.float64)
    output_scores = 0.500 + 0.499 * normalized
    for index in range(1, len(output_scores)):
        output_scores[index] = min(output_scores[index], output_scores[index - 1] - 0.000001)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=["candidate_id", "rank", "score", "reasoning"],
        )
        writer.writeheader()
        for index, row in enumerate(ranked):
            row["rank"] = index + 1
            writer.writerow(
                {
                    "candidate_id": row["candidate_id"],
                    "rank": row["rank"],
                    "score": f"{output_scores[index]:.6f}",
                    "reasoning": reasoning_for_row(row, row["rank"]),
                }
            )
    return output
