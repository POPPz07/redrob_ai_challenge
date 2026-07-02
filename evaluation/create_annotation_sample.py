"""Create a deterministic, model-blinded candidate annotation sample."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CANDIDATES = "India_runs_data_and_ai_challenge/candidates.jsonl"
DEFAULT_FEATURES = "artifacts/candidates_features.parquet"
DEFAULT_SUBMISSION = "submission.csv"
DEFAULT_OUTPUT_DIR = "outputs/annotation_audit"
SAMPLE_SIZE = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    parser.add_argument("--features", default=DEFAULT_FEATURES)
    parser.add_argument("--submission", default=DEFAULT_SUBMISSION)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def stable_order(candidate_id: str, namespace: str) -> str:
    return hashlib.sha256(f"bug-solvers-blind-v1|{namespace}|{candidate_id}".encode()).hexdigest()


def choose(frame: pd.DataFrame, mask: pd.Series, count: int, used: set[str], namespace: str) -> list[str]:
    eligible = frame.loc[mask & ~frame.candidate_id.isin(used), "candidate_id"].tolist()
    eligible.sort(key=lambda candidate_id: stable_order(candidate_id, namespace))
    picked = eligible[:count]
    used.update(picked)
    return picked


def select_candidates(features: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    used: set[str] = set()
    strata: dict[str, str] = {}

    def add(name: str, ids: list[str]) -> None:
        for candidate_id in ids:
            strata[candidate_id] = name

    # Include every semantically verified rare narrative so the suspected blind spot is testable.
    ids = features.loc[features.rare_expert_narrative >= 1, "candidate_id"].tolist()
    ids.sort(key=lambda candidate_id: stable_order(candidate_id, "rare_expert"))
    used.update(ids)
    add("rare_expert_census", ids)

    risk_sorted = features.loc[
        (features.honeypot_risk >= 0.20) & ~features.candidate_id.isin(used)
    ].sort_values(["honeypot_risk", "candidate_id"], ascending=[False, True])
    ids = risk_sorted.head(31).candidate_id.tolist()
    used.update(ids)
    add("high_consistency_risk", ids)

    ids = choose(
        features,
        (features.weighted_formula_score >= features.weighted_formula_score.quantile(0.985))
        & (features.honeypot_risk < 0.20),
        45,
        used,
        "strong_nonrare",
    )
    add("strong_nonrare", ids)

    ids = choose(
        features,
        (features.retrieval_ranking_depth >= 0.55)
        & (features.production_ml_depth >= 0.35)
        & (features.weighted_formula_score < features.weighted_formula_score.quantile(0.985)),
        35,
        used,
        "hard_retrieval_nearmatch",
    )
    add("hard_retrieval_nearmatch", ids)

    ids = choose(
        features,
        (features.technical_title >= 0.5)
        & ((features.production_ml_depth >= 0.20) | (features.python_engineering_fit >= 0.45))
        & (features.weighted_formula_score.between(0.25, 0.52)),
        30,
        used,
        "adjacent_technical",
    )
    add("adjacent_technical", ids)

    ids = choose(
        features,
        (features.weighted_formula_score.between(0.12, 0.32))
        & (features.honeypot_risk < 0.20),
        20,
        used,
        "moderate_random",
    )
    add("moderate_random", ids)

    remaining = SAMPLE_SIZE - len(used)
    ids = choose(
        features,
        (features.weighted_formula_score <= features.weighted_formula_score.quantile(0.15)),
        remaining,
        used,
        "low_relevance_control",
    )
    add("low_relevance_control", ids)

    if len(used) != SAMPLE_SIZE:
        raise RuntimeError(f"Expected {SAMPLE_SIZE} unique profiles, selected {len(used)}")

    ordered = sorted(used, key=lambda candidate_id: stable_order(candidate_id, "workbook_order"))
    return ordered, strata


def compact_text(value: object) -> str:
    return " ".join(str(value or "").split())


def format_career(candidate: dict) -> str:
    rows = []
    for job in candidate.get("career_history", []):
        current = "current" if job.get("is_current") else f"ended {job.get('end_date') or 'unknown'}"
        rows.append(
            f"{job.get('title', '')} at {job.get('company', '')} | {job.get('industry', '')} | "
            f"{job.get('duration_months', 0)} months, {current}\n{compact_text(job.get('description'))}"
        )
    return "\n\n".join(rows)


def format_skills(candidate: dict) -> str:
    assessments = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})
    rows = []
    for skill in candidate.get("skills", []):
        assessment = assessments.get(skill.get("name"))
        assessment_text = f", assessment {assessment:g}/100" if isinstance(assessment, (int, float)) else ""
        rows.append(
            f"{skill.get('name', '')}: {skill.get('proficiency', 'unknown')}, "
            f"{skill.get('duration_months', 0)} months, {skill.get('endorsements', 0)} endorsements{assessment_text}"
        )
    return "; ".join(rows)


def format_education(candidate: dict) -> str:
    return "; ".join(
        f"{row.get('degree', '')} in {row.get('field_of_study', '')}, {row.get('institution', '')} "
        f"({row.get('end_year', '')}, {row.get('tier', 'unknown')})"
        for row in candidate.get("education", [])
    )


def candidate_row(profile_key: str, candidate: dict) -> dict:
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    salary = signals.get("expected_salary_range_inr_lpa", {})
    return {
        "profile_key": profile_key,
        "current_title": compact_text(profile.get("current_title")),
        "current_company": compact_text(profile.get("current_company")),
        "years_experience": profile.get("years_of_experience"),
        "location": f"{profile.get('location', '')}, {profile.get('country', '')}".strip(", "),
        "industry": compact_text(profile.get("current_industry")),
        "headline": compact_text(profile.get("headline")),
        "summary": compact_text(profile.get("summary")),
        "career_history": format_career(candidate),
        "skills_and_evidence": format_skills(candidate),
        "education": format_education(candidate),
        "availability": (
            f"open_to_work={signals.get('open_to_work_flag')}; last_active={signals.get('last_active_date')}; "
            f"response_rate={signals.get('recruiter_response_rate')}; avg_response_hours={signals.get('avg_response_time_hours')}; "
            f"notice_days={signals.get('notice_period_days')}; work_mode={signals.get('preferred_work_mode')}; "
            f"willing_to_relocate={signals.get('willing_to_relocate')}"
        ),
        "market_and_trust_signals": (
            f"profile_complete={signals.get('profile_completeness_score')}; github={signals.get('github_activity_score')}; "
            f"saved_30d={signals.get('saved_by_recruiters_30d')}; search_appearances_30d={signals.get('search_appearance_30d')}; "
            f"interview_completion={signals.get('interview_completion_rate')}; offer_acceptance={signals.get('offer_acceptance_rate')}; "
            f"verified_email={signals.get('verified_email')}; verified_phone={signals.get('verified_phone')}; "
            f"linkedin={signals.get('linkedin_connected')}; expected_salary_lpa={salary.get('min')}-{salary.get('max')}"
        ),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features = pd.read_parquet(args.features)
    selected, strata = select_candidates(features)
    selected_set = set(selected)
    feature_rows = features.set_index("candidate_id")

    rank_map: dict[str, int] = {}
    submission_path = Path(args.submission)
    if submission_path.exists():
        submission = pd.read_csv(submission_path)
        rank_map = dict(zip(submission.candidate_id, submission["rank"], strict=False))

    profile_keys = {candidate_id: f"PROFILE-{index:03d}" for index, candidate_id in enumerate(selected, 1)}
    evaluation_splits: dict[str, str] = {}
    for stratum in sorted(set(strata.values())):
        members = [candidate_id for candidate_id in selected if strata[candidate_id] == stratum]
        members.sort(key=lambda candidate_id: stable_order(candidate_id, f"split|{stratum}"))
        development_count = round(0.60 * len(members))
        for index, candidate_id in enumerate(members):
            evaluation_splits[candidate_id] = "development" if index < development_count else "holdout"
    candidate_rows: dict[str, dict] = {}
    with Path(args.candidates).open("r", encoding="utf-8") as source:
        for line in source:
            candidate = json.loads(line)
            candidate_id = candidate["candidate_id"]
            if candidate_id in selected_set:
                candidate_rows[candidate_id] = candidate_row(profile_keys[candidate_id], candidate)

    missing = selected_set - candidate_rows.keys()
    if missing:
        raise RuntimeError(f"Selected candidates missing from JSONL: {sorted(missing)[:5]}")

    pack = [candidate_rows[candidate_id] for candidate_id in selected]
    (output_dir / "annotation_profiles.json").write_text(
        json.dumps(pack, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    key_columns = [
        "profile_key", "candidate_id", "stratum", "evaluation_split", "current_submission_rank",
        "weighted_formula_score", "career_system_fit", "retrieval_ranking_depth",
        "production_ml_depth", "evaluation_experimentation_fit", "rare_expert_narrative",
        "honeypot_risk", "consistency_score",
    ]
    with (output_dir / "annotation_key.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=key_columns)
        writer.writeheader()
        for candidate_id in selected:
            row = feature_rows.loc[candidate_id]
            writer.writerow({
                "profile_key": profile_keys[candidate_id],
                "candidate_id": candidate_id,
                "stratum": strata[candidate_id],
                "evaluation_split": evaluation_splits[candidate_id],
                "current_submission_rank": rank_map.get(candidate_id, ""),
                "weighted_formula_score": row.weighted_formula_score,
                "career_system_fit": row.career_system_fit,
                "retrieval_ranking_depth": row.retrieval_ranking_depth,
                "production_ml_depth": row.production_ml_depth,
                "evaluation_experimentation_fit": row.evaluation_experimentation_fit,
                "rare_expert_narrative": row.rare_expert_narrative,
                "honeypot_risk": row.honeypot_risk,
                "consistency_score": row.consistency_score,
            })

    counts = pd.Series(strata).value_counts().sort_index().to_dict()
    split_counts = pd.Series(evaluation_splits).value_counts().sort_index().to_dict()
    print(json.dumps({"sample_size": len(pack), "strata": counts, "splits": split_counts, "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()