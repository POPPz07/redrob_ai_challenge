from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Redrob ranker sanity checks.")
    parser.add_argument("--submission", default="submission.csv")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--candidates", default="India_runs_data_and_ai_challenge/candidates.jsonl")
    parser.add_argument("--skip-determinism", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_candidate_ids(candidate_path: Path, selected_ids: set[str]) -> None:
    found_ids: set[str] = set()
    with candidate_path.open("r", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                candidate_id = json.loads(line).get("candidate_id")
                if candidate_id in selected_ids:
                    found_ids.add(candidate_id)
    assert found_ids == selected_ids, f"candidate IDs missing from source: {sorted(selected_ids - found_ids)}"


def verify_no_network_imports() -> None:
    forbidden = {"requests", "httpx", "socket", "urllib", "aiohttp"}
    files = [Path("rank.py"), *Path("redrob_ranker").glob("*.py")]
    found: set[str] = set()
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                found.add(node.module.split(".")[0])
    blocked = forbidden & found
    assert not blocked, f"network-capable imports in ranking path: {sorted(blocked)}"

def main() -> None:
    args = parse_args()
    submission = Path(args.submission)
    validator = Path("India_runs_data_and_ai_challenge/validate_submission.py")
    subprocess.run([sys.executable, str(validator), str(submission)], check=True)

    rows = read_rows(submission)
    assert len(rows) == 100, "expected 100 data rows"
    scores = [float(row["score"]) for row in rows]
    assert all(scores[i] > scores[i + 1] for i in range(len(scores) - 1)), "scores must be strictly descending"
    assert len({row["candidate_id"] for row in rows}) == 100, "candidate IDs must be unique"
    assert [int(row["rank"]) for row in rows] == list(range(1, 101)), "ranks must be 1..100"
    assert all(len(row["reasoning"].strip()) > 25 for row in rows), "reasoning should be non-empty"

    selected_ids = {row["candidate_id"] for row in rows}
    verify_candidate_ids(Path(args.candidates), selected_ids)
    verify_no_network_imports()

    features = pd.read_parquet(Path(args.artifacts_dir) / "candidates_features.parquet")
    selected = features[features["candidate_id"].isin(selected_ids)].set_index("candidate_id")
    assert len(selected) == 100, "all selected candidates must have feature rows"
    suspicious = selected[
        (selected["honeypot_risk"] >= 0.30) & (selected["consistency_score"] < 0.65)
    ]
    assert suspicious.empty, f"high-risk inconsistent candidates selected: {suspicious.index.tolist()}"
    expert_zero = selected[selected["suspicious_expert_skill_count"] >= 3]
    assert expert_zero.empty, (
        "candidates with multiple zero-duration expert skills selected: "
        f"{expert_zero.index.tolist()}"
    )
    for row in rows:
        title = str(selected.loc[row["candidate_id"], "current_title"])
        assert title.lower() in row["reasoning"].lower(), f"title missing from reasoning: {row['candidate_id']}"

    risky = {
        "CAND_0016000",
        "CAND_0046649",
        "CAND_0056983",
        "CAND_0060642",
        "CAND_0061722",
    }
    overlap = risky & {row["candidate_id"] for row in rows[:50]}
    assert not overlap, f"obvious honeypot-risk candidates in top 50: {sorted(overlap)}"

    sample = pd.read_csv("India_runs_data_and_ai_challenge/sample_submission.csv")
    nontechnical_prefixes = (
        "HR Manager",
        "Marketing Manager",
        "Content Writer",
        "Graphic Designer",
        "Mechanical Engineer",
        "Civil Engineer",
        "Accountant",
        "Sales Executive",
        "Operations Manager",
        "Customer Support",
    )
    sample_traps = set(
        sample[sample["reasoning"].str.startswith(nontechnical_prefixes)]["candidate_id"]
    )
    sample_overlap = sample_traps & {row["candidate_id"] for row in rows[:50]}
    assert not sample_overlap, f"sample keyword traps in top 50: {sorted(sample_overlap)}"
    assert "CAND_0000031" in selected_ids, "clear sample recommendation engineer missing from top 100"

    if not args.skip_determinism:
        before = file_hash(submission)
        temporary = submission.with_suffix(".determinism.csv")
        offline_env = os.environ.copy()
        offline_env.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
            }
        )
        started = time.perf_counter()
        subprocess.run(
            [
                sys.executable,
                "rank.py",
                "--candidates",
                args.candidates,
                "--artifacts-dir",
                args.artifacts_dir,
                "--out",
                str(temporary),
            ],
            check=True,
            env=offline_env,
        )
        elapsed = time.perf_counter() - started
        after = file_hash(temporary)
        temporary.unlink(missing_ok=True)
        assert before == after, "determinism check failed"
        assert elapsed < 300, f"ranking exceeded 5 minutes: {elapsed:.1f}s"
        print(f"deterministic offline rerun: {elapsed:.2f}s")

    print("All checks passed.")


if __name__ == "__main__":
    main()


