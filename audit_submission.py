from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a fact-level submission audit report.")
    parser.add_argument("--submission", default="submission.csv")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--out", default="artifacts/submission_audit.csv")
    args = parser.parse_args()

    submission = pd.read_csv(args.submission)
    artifacts = Path(args.artifacts_dir)
    features = pd.read_parquet(artifacts / "candidates_features.parquet")
    texts = pd.read_parquet(artifacts / "candidate_text_views.parquet")
    report = submission.merge(features, on="candidate_id", how="left", validate="one_to_one")
    report = report.merge(
        texts[["candidate_id", "evidence_text"]],
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    columns = [
        "rank",
        "candidate_id",
        "score",
        "current_title",
        "current_company",
        "years_experience",
        "location",
        "career_system_fit",
        "retrieval_ranking_depth",
        "production_ml_depth",
        "evaluation_experimentation_fit",
        "python_engineering_fit",
        "skill_trust_score",
        "consistency_score",
        "honeypot_risk",
        "keyword_stuffing_penalty",
        "services_penalty",
        "reason_concern",
        "evidence_text",
        "reasoning",
    ]
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    report[columns].sort_values("rank").to_csv(output, index=False)
    print(f"Wrote {len(report)} audited candidates to {output}")
    means = report.nsmallest(10, "rank")[[
        "career_system_fit",
        "retrieval_ranking_depth",
        "production_ml_depth",
        "evaluation_experimentation_fit",
        "consistency_score",
    ]].mean().round(3).to_dict()
    print("Top-10 mean signals:", means)


if __name__ == "__main__":
    main()
