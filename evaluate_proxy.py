from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a submission against the dataset's empirical archetype bands."
    )
    parser.add_argument(
        "--candidates",
        default="India_runs_data_and_ai_challenge/candidates.jsonl",
    )
    parser.add_argument("--submission", default="submission.csv")
    parser.add_argument("--artifacts-dir", default="artifacts")
    return parser.parse_args()


def normalize_description(value: object) -> str:
    return " ".join(str(value or "").split())


def frequency_band(frequency: int) -> int:
    """Map the six sharply separated narrative-frequency bands to 0..5."""
    if frequency > 20_000:
        return 0
    if frequency > 5_000:
        return 1
    if frequency > 1_000:
        return 2
    if frequency > 100:
        return 3
    if frequency > 20:
        return 4
    return 5


def discounted_gain(candidate_ids: list[str], relevance: dict[str, int], k: int) -> float:
    return sum(
        ((2 ** relevance[candidate_id]) - 1) / math.log2(index + 2)
        for index, candidate_id in enumerate(candidate_ids[:k])
    )


def average_precision(candidate_ids: list[str], relevance: dict[str, int]) -> float:
    relevant_total = sum(value >= 3 for value in relevance.values())
    if not relevant_total:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for index, candidate_id in enumerate(candidate_ids, 1):
        if relevance[candidate_id] >= 3:
            hits += 1
            precision_sum += hits / index
    return precision_sum / relevant_total


def main() -> None:
    args = parse_args()
    candidates_path = Path(args.candidates)
    description_counts: Counter[str] = Counter()
    candidate_descriptions: dict[str, list[str]] = {}

    with candidates_path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            candidate = json.loads(line)
            descriptions = [
                normalize_description(job.get("description"))
                for job in candidate.get("career_history", [])
            ]
            descriptions = [description for description in descriptions if description]
            candidate_descriptions[candidate["candidate_id"]] = descriptions
            description_counts.update(descriptions)

    relevance = {
        candidate_id: max(
            (frequency_band(description_counts[description]) for description in descriptions),
            default=0,
        )
        for candidate_id, descriptions in candidate_descriptions.items()
    }

    features_path = Path(args.artifacts_dir) / "candidates_features.parquet"
    if features_path.exists():
        features = pd.read_parquet(
            features_path,
            columns=["candidate_id", "honeypot_risk", "consistency_score"],
        )
        for row in features.itertuples(index=False):
            if row.honeypot_risk >= 0.30 and row.consistency_score < 0.65:
                relevance[row.candidate_id] = 0

    with Path(args.submission).open("r", encoding="utf-8", newline="") as source:
        ranked_ids = [row["candidate_id"] for row in csv.DictReader(source)]

    missing = [candidate_id for candidate_id in ranked_ids if candidate_id not in relevance]
    if missing:
        raise ValueError(f"Submission IDs missing from candidates: {missing[:5]}")

    ideal_ids = sorted(relevance, key=lambda candidate_id: (-relevance[candidate_id], candidate_id))
    ndcg: dict[int, float] = {}
    for k in (10, 50, 100):
        ideal = discounted_gain(ideal_ids, relevance, k)
        ndcg[k] = discounted_gain(ranked_ids, relevance, k) / ideal if ideal else 0.0

    map_score = average_precision(ranked_ids, relevance)
    precision_at_10 = sum(relevance[candidate_id] >= 3 for candidate_id in ranked_ids[:10]) / 10
    result = {
        "warning": "Local archetype proxy only; this is not organizer ground truth.",
        "description_templates": len(description_counts),
        "candidate_count": len(relevance),
        "ndcg_at_10": round(ndcg[10], 6),
        "ndcg_at_50": round(ndcg[50], 6),
        "ndcg_at_100": round(ndcg[100], 6),
        "map_tier_3_plus": round(map_score, 6),
        "precision_at_10_tier_3_plus": round(precision_at_10, 6),
        "proxy_composite": round(
            0.50 * ndcg[10]
            + 0.30 * ndcg[50]
            + 0.15 * map_score
            + 0.05 * precision_at_10,
            6,
        ),
        "top_10_tiers": Counter(relevance[candidate_id] for candidate_id in ranked_ids[:10]),
        "top_50_tiers": Counter(relevance[candidate_id] for candidate_id in ranked_ids[:50]),
        "top_100_tiers": Counter(relevance[candidate_id] for candidate_id in ranked_ids[:100]),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
