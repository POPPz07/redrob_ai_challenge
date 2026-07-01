from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from redrob_ranker.neural_retrieval import MODEL_PATH


PRODUCTION_ARTIFACTS = (
    "candidate_text_views.parquet",
    "candidates_features.parquet",
    "tfidf_vectorizer.joblib",
    "tfidf_matrix.npz",
    "dense_svd.joblib",
    "dense_embeddings.npy",
    "dense_index.faiss",
    "dense_query_embedding.npy",
    "ranker_model.joblib",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce the full neural Redrob submission from candidate JSONL."
    )
    parser.add_argument(
        "--candidates", default="India_runs_data_and_ai_challenge/candidates.jsonl"
    )
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--out", default="submission.csv")
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    args = parser.parse_args()

    candidates = Path(args.candidates)
    artifacts = Path(args.artifacts_dir)
    model_path = Path(args.model_path)
    if not candidates.exists():
        raise FileNotFoundError(f"Candidate JSONL not found: {candidates}")

    missing = [name for name in PRODUCTION_ARTIFACTS if not (artifacts / name).exists()]
    if missing:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Local embedding model not found at {model_path}. "
                "Run `python download_model.py` during setup, then retry."
            )
        subprocess.run(
            [
                sys.executable,
                "precompute.py",
                "--candidates",
                str(candidates),
                "--artifacts-dir",
                str(artifacts),
                "--model-path",
                str(model_path),
            ],
            check=True,
        )

    subprocess.run(
        [
            sys.executable,
            "rank.py",
            "--candidates",
            str(candidates),
            "--artifacts-dir",
            str(artifacts),
            "--out",
            args.out,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
