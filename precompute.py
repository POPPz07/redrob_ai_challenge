from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from redrob_ranker.features import ASPECT_QUERIES, clean_text, extract_features
from redrob_ranker.neural_retrieval import MODEL_PATH, build_neural_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute Redrob ranker artifacts.")
    parser.add_argument("--candidates", default="India_runs_data_and_ai_challenge/candidates.jsonl")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--max-features", type=int, default=70000)
    parser.add_argument("--svd-dim", type=int, default=192)
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--skip-neural", action="store_true")
    parser.add_argument(
        "--features-only",
        action="store_true",
        help="Refresh feature Parquet and ranker model while preserving retrieval artifacts.",
    )
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    return parser.parse_args()


def load_candidates(path: Path) -> pd.DataFrame:
    description_frequencies: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            candidate = json.loads(line)
            description_frequencies.update(
                description
                for description in (
                    clean_text(job.get("description"))
                    for job in candidate.get("career_history", [])
                )
                if description
            )
    print(
        f"found {len(description_frequencies):,} unique career narratives "
        f"across {sum(description_frequencies.values()):,} jobs"
    )

    rows = []
    started = time.perf_counter()
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                rows.append(
                    extract_features(
                        json.loads(line),
                        description_frequencies=description_frequencies,
                    )
                )
            if i % 25000 == 0:
                print(f"parsed {i:,} candidates in {time.perf_counter() - started:.1f}s")
    return pd.DataFrame(rows)


def existing_neural_metadata(artifacts_dir: Path) -> dict[str, object]:
    index_path = artifacts_dir / "dense_index.faiss"
    query_path = artifacts_dir / "dense_query_embedding.npy"
    if not index_path.exists() or not query_path.exists():
        return {}
    try:
        import faiss

        index = faiss.read_index(str(index_path))
        return {
            "faiss_available": True,
            "faiss_index_type": type(index).__name__,
            "faiss_count": int(index.ntotal),
            "faiss_dimension": int(index.d),
            "faiss_index_bytes": int(index_path.stat().st_size),
            "neural_model": "BAAI/bge-small-en-v1.5",
        }
    except Exception as exc:
        print(f"existing FAISS metadata unavailable: {exc}")
        return {}


def main() -> None:
    args = parse_args()
    candidates_path = Path(args.candidates)
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    df = load_candidates(candidates_path)
    print(f"loaded {len(df):,} candidates")

    text_cols = ["candidate_id", "career_text", "profile_text", "skills_text", "evidence_text", "all_text"]
    feature_keep = [
        c
        for c in df.columns
        if c not in {"career_text", "profile_text", "skills_text", "evidence_text", "all_text"}
    ]
    df[text_cols].to_parquet(artifacts_dir / "candidate_text_views.parquet", index=False)
    df[feature_keep].to_parquet(artifacts_dir / "candidates_features.parquet", index=False)

    if args.features_only:
        required = [
            "tfidf_vectorizer.joblib",
            "tfidf_matrix.npz",
            "dense_svd.joblib",
            "dense_embeddings.npy",
        ]
        missing = [name for name in required if not (artifacts_dir / name).exists()]
        if missing:
            raise FileNotFoundError(
                f"Cannot use --features-only; missing retrieval artifacts: {missing}"
            )
        if not args.skip_model:
            subprocess.run(
                [
                    sys.executable,
                    "retrain_ranker.py",
                    "--artifacts-dir",
                    str(artifacts_dir),
                ],
                check=True,
            )
        metadata_path = artifacts_dir / "precompute_metadata.json"
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists()
            else {}
        )
        metadata.update(
            {
                "candidate_count": int(len(df)),
                "feature_refresh_only": True,
                "feature_refresh_seconds": round(time.perf_counter() - started, 3),
            }
        )
        metadata.update(existing_neural_metadata(artifacts_dir))
        if (artifacts_dir / "ranker_model.joblib").exists():
            payload = joblib.load(artifacts_dir / "ranker_model.joblib")
            metadata["model_available"] = True
            metadata["feature_columns"] = payload["feature_cols"]
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps(metadata, indent=2))
        return

    retrieval_text = (
        df["evidence_text"].fillna("")
        + " "
        + df["career_text"].fillna("")
        + " "
        + df["profile_text"].fillna("")
        + " "
        + df["skills_text"].fillna("")
    ).tolist()
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.90,
        max_features=args.max_features,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    tfidf = vectorizer.fit_transform(retrieval_text)
    sparse.save_npz(artifacts_dir / "tfidf_matrix.npz", tfidf)
    joblib.dump(vectorizer, artifacts_dir / "tfidf_vectorizer.joblib")

    svd_dim = min(args.svd_dim, max(2, tfidf.shape[1] - 1))
    svd = TruncatedSVD(n_components=svd_dim, random_state=42)
    dense = svd.fit_transform(tfidf)
    dense = normalize(dense, norm="l2", copy=False).astype(np.float32)
    np.save(artifacts_dir / "dense_embeddings.npy", dense)
    joblib.dump(svd, artifacts_dir / "dense_svd.joblib")

    model_ok = False
    feature_cols: list[str] = []

    metadata = {
        "candidate_count": int(len(df)),
        "tfidf_shape": list(tfidf.shape),
        "svd_dim": int(svd_dim),
        "model_available": model_ok,
        "feature_columns": feature_cols,
        "aspect_queries": ASPECT_QUERIES,
        "faiss_available": False,
        "colbert_available": False,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    if not args.skip_neural:
        neural_metadata = build_neural_artifacts(
            df[text_cols],
            artifacts_dir,
            " ".join(ASPECT_QUERIES.values()),
            model_path=Path(args.model_path),
            batch_size=args.embedding_batch_size,
        )
        metadata.update(neural_metadata)
    else:
        metadata.update(existing_neural_metadata(artifacts_dir))
    if not args.skip_model:
        subprocess.run(
            [
                sys.executable,
                "retrain_ranker.py",
                "--artifacts-dir",
                str(artifacts_dir),
            ],
            check=True,
        )
        payload = joblib.load(artifacts_dir / "ranker_model.joblib")
        model_ok = True
        feature_cols = payload["feature_cols"]
        metadata["model_available"] = model_ok
        metadata["feature_columns"] = feature_cols
    (artifacts_dir / "precompute_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
