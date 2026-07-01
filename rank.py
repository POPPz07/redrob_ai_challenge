from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize

from redrob_ranker.features import ASPECT_QUERIES, NUMERIC_FEATURES, clamp, reasoning_for_row, stable_minmax
from redrob_ranker.late_interaction import USE_COLBERT, score_late_interaction


DEFAULT_QUERY = " ".join(ASPECT_QUERIES.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank Redrob candidates.")
    parser.add_argument("--candidates", default="India_runs_data_and_ai_challenge/candidates.jsonl", help="Kept for reproduce-command compatibility; artifacts are used for ranking.")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--out", default="submission.csv")
    parser.add_argument("--pool-size", type=int, default=12000)
    parser.add_argument("--top-n", type=int, default=100)
    return parser.parse_args()


def top_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(scores))
    if k <= 0:
        return np.array([], dtype=np.int64)
    part = np.argpartition(-scores, k - 1)[:k]
    return part[np.argsort(-scores[part], kind="mergesort")]


def load_artifacts(artifacts_dir: Path):
    features = pd.read_parquet(artifacts_dir / "candidates_features.parquet")
    texts = pd.read_parquet(artifacts_dir / "candidate_text_views.parquet")
    vectorizer = joblib.load(artifacts_dir / "tfidf_vectorizer.joblib")
    tfidf = sparse.load_npz(artifacts_dir / "tfidf_matrix.npz")
    svd = joblib.load(artifacts_dir / "dense_svd.joblib")
    dense = np.load(artifacts_dir / "dense_embeddings.npy", mmap_mode="r")
    neural_index = None
    neural_query = None
    index_path = artifacts_dir / "dense_index.faiss"
    query_path = artifacts_dir / "dense_query_embedding.npy"
    if index_path.exists() and query_path.exists():
        try:
            import faiss

            neural_index = faiss.read_index(str(index_path))
            neural_query = np.load(query_path).astype(np.float32)
            if neural_index.ntotal != len(features):
                raise ValueError("FAISS candidate count does not match feature rows")
        except Exception as exc:
            print(f"FAISS fallback used: {exc}")
            neural_index = None
            neural_query = None
    return features, texts, vectorizer, tfidf, svd, dense, neural_index, neural_query


def retrieve_pool(
    features: pd.DataFrame,
    vectorizer,
    tfidf,
    svd,
    dense,
    pool_size: int,
    neural_index=None,
    neural_query=None,
) -> tuple[np.ndarray, dict[int, dict[str, float]]]:
    query_vec = vectorizer.transform([DEFAULT_QUERY])
    sparse_scores = (tfidf @ query_vec.T).toarray().ravel()
    query_dense = normalize(svd.transform(query_vec), norm="l2", copy=False).astype(np.float32)
    svd_scores = np.asarray(dense @ query_dense.T).ravel()
    neural_scores: dict[int, float] = {}
    neural_ids = np.array([], dtype=np.int64)
    if neural_index is not None and neural_query is not None:
        distances, indices = neural_index.search(
            neural_query, min(pool_size // 2, neural_index.ntotal)
        )
        neural_ids = indices[0][indices[0] >= 0].astype(np.int64)
        neural_scores = {
            int(idx): float(score)
            for idx, score in zip(indices[0], distances[0])
            if idx >= 0
        }

    rule = (
        0.28 * features["career_system_fit"].to_numpy()
        + 0.22 * features["retrieval_ranking_depth"].to_numpy()
        + 0.16 * features["production_ml_depth"].to_numpy()
        + 0.12 * features["evaluation_experimentation_fit"].to_numpy()
        + 0.08 * features["high_signal_title"].to_numpy()
        + 0.08 * features["skill_trust_score"].to_numpy()
        + 0.06 * features["weighted_formula_score"].to_numpy()
        + 0.16 * features["rare_expert_narrative"].to_numpy()
        + 0.04 * features["career_narrative_strength"].to_numpy()
    )
    rule = rule - 0.20 * features["honeypot_risk"].to_numpy() - 0.12 * features["keyword_stuffing_penalty"].to_numpy()

    title_mask = (
        (features["high_signal_title"].to_numpy() > 0)
        | ((features["technical_title"].to_numpy() > 0) & (features["retrieval_ranking_depth"].to_numpy() > 0.25))
        | ((features["career_system_fit"].to_numpy() > 0.55) & (features["production_ml_depth"].to_numpy() > 0.35))
        | (features["rare_expert_narrative"].to_numpy() > 0)
    )
    candidates: set[int] = set(top_indices(sparse_scores, pool_size // 2))
    candidates.update(top_indices(svd_scores, pool_size // 2))
    candidates.update(neural_ids.tolist())
    candidates.update(top_indices(rule, pool_size // 2))
    candidates.update(np.flatnonzero(title_mask).tolist())
    pool = np.array(sorted(candidates), dtype=np.int64)
    if len(pool) > pool_size:
        combined = 0.36 * np.asarray(stable_minmax(sparse_scores[pool].tolist()))
        combined = np.asarray(combined) + 0.15 * np.asarray(stable_minmax(svd_scores[pool].tolist()))
        pool_neural = np.array([neural_scores.get(int(i), 0.0) for i in pool])
        combined = combined + 0.15 * np.asarray(stable_minmax(pool_neural.tolist()))
        combined = combined + 0.34 * np.asarray(stable_minmax(rule[pool].tolist()))
        keep = top_indices(combined, pool_size)
        pool = pool[keep]

    score_lookup = {
        int(i): {
            "sparse_score": float(sparse_scores[i]),
            "dense_score": float(neural_scores.get(int(i), svd_scores[i])),
            "svd_score": float(svd_scores[i]),
            "rule_recall_score": float(rule[i]),
        }
        for i in pool
    }
    return pool, score_lookup


def predict_scores(pool_df: pd.DataFrame, artifacts_dir: Path) -> np.ndarray:
    base = pool_df["weighted_formula_score"].astype(float).to_numpy()
    feature_cols = list(NUMERIC_FEATURES) + [
        "sparse_score",
        "dense_score",
        "rule_recall_score",
        "colbert_maxsim_score",
        "colbert_retrieval_score",
        "colbert_ranking_score",
        "colbert_eval_score",
        "colbert_production_ml_score",
        "colbert_python_score",
    ]
    model_path = artifacts_dir / "ranker_model.joblib"
    if model_path.exists():
        try:
            payload = joblib.load(model_path)
            feature_cols = payload["feature_cols"]
            model = payload["model"]
            X = pool_df.reindex(columns=feature_cols, fill_value=0.0).astype(np.float32).to_numpy()
            model_scores = np.asarray(model.predict(X), dtype=np.float32)
            model_scores = np.asarray(stable_minmax(model_scores.tolist()))
            base_norm = np.asarray(stable_minmax(base.tolist()))
            return 0.58 * base_norm + 0.42 * model_scores
        except Exception as exc:
            print(f"model fallback used: {exc}")
    return np.asarray(stable_minmax(base.tolist()))


def apply_final_gates(df: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
    gated = scores.astype(np.float64).copy()
    gated += 0.030 * df["sparse_score_norm"].to_numpy()
    gated += 0.025 * df["dense_score_norm"].to_numpy()
    gated += 0.055 * df["colbert_maxsim_score"].to_numpy()
    gated += 0.025 * df["production_ml_depth"].to_numpy()
    gated += 0.015 * df["product_company_fit"].to_numpy()
    gated += 0.010 * df["experience_fit"].to_numpy()
    gated += 0.150 * df["rare_expert_narrative"].to_numpy()
    gated += 0.025 * df["current_rare_expert_narrative"].to_numpy()
    gated += 0.025 * df["career_narrative_strength"].to_numpy()
    gated += 0.015 * df["high_relevance_narrative_count"].to_numpy()
    gated -= 0.035 * df["keyword_stuffing_penalty"].to_numpy()
    gated -= 0.025 * df["title_mismatch_penalty"].to_numpy()

    suspicious_consistency = (
        (df["honeypot_risk"].to_numpy() >= 0.30)
        & (df["consistency_score"].to_numpy() < 0.65)
    )
    gated *= np.where(suspicious_consistency, 0.05, 1.0)
    gated *= np.where(df["honeypot_risk"].to_numpy() >= 0.72, 0.05, 1.0)
    gated *= np.where(df["nontechnical_ai_stuffer"].to_numpy() >= 1.0, 0.35, 1.0)
    gated *= np.where(df["services_penalty"].to_numpy() >= 1.0, 0.45, 1.0)
    gated *= np.where(df["no_production_evidence"].to_numpy() >= 1.0, 0.60, 1.0)
    gated *= np.where(df["outside_india_penalty"].to_numpy() >= 1.0, 0.55, 1.0)
    gated *= np.where(df["inactive_low_response_penalty"].to_numpy() >= 1.0, 0.50, 1.0)
    # Preserve score separation above 1.0. Clipping here creates large tie groups
    # and silently turns candidate_id into the primary top-rank signal.
    return gated


def calibrated_output_scores(scores: np.ndarray) -> np.ndarray:
    normalized = np.asarray(stable_minmax(scores.tolist()), dtype=np.float64)
    calibrated = 0.500 + 0.499 * normalized
    # Rounded CSV values must remain strictly descending for the validator.
    for idx in range(1, len(calibrated)):
        calibrated[idx] = min(calibrated[idx], calibrated[idx - 1] - 0.000001)
    return calibrated


def select_top_candidates(pool_features: pd.DataFrame, top_n: int) -> pd.DataFrame:
    return pool_features.sort_values(
        ["final_score", "candidate_id"],
        ascending=[False, True],
        kind="mergesort",
    ).head(top_n).copy()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    artifacts_dir = Path(args.artifacts_dir)
    required = [
        artifacts_dir / "candidates_features.parquet",
        artifacts_dir / "candidate_text_views.parquet",
        artifacts_dir / "tfidf_vectorizer.joblib",
        artifacts_dir / "tfidf_matrix.npz",
        artifacts_dir / "dense_svd.joblib",
        artifacts_dir / "dense_embeddings.npy",
        artifacts_dir / "ranker_model.joblib",
    ]
    if not all(path.exists() for path in required):
        print("base artifacts missing; running reproducible CPU precompute", flush=True)
        subprocess.run(
            [
                sys.executable,
                "precompute.py",
                "--candidates",
                args.candidates,
                "--artifacts-dir",
                args.artifacts_dir,
                "--skip-neural",
            ],
            check=True,
        )
    features, texts, vectorizer, tfidf, svd, dense, neural_index, neural_query = load_artifacts(artifacts_dir)
    pool, retrieval_scores = retrieve_pool(
        features,
        vectorizer,
        tfidf,
        svd,
        dense,
        args.pool_size,
        neural_index,
        neural_query,
    )
    pool_features = features.iloc[pool].copy().reset_index(drop=True)
    pool_texts = texts.iloc[pool].copy().reset_index(drop=True)

    retrieval_df = pd.DataFrame([retrieval_scores[int(row_idx)] for row_idx in pool])
    pool_features = pd.concat(
        [pool_features.reset_index(drop=True), retrieval_df.reset_index(drop=True)],
        axis=1,
    )
    pool_features["sparse_score_norm"] = stable_minmax(pool_features["sparse_score"].astype(float).tolist())
    pool_features["dense_score_norm"] = stable_minmax(pool_features["dense_score"].astype(float).tolist())

    if USE_COLBERT:
        late_rows = [
            score_late_interaction(row.evidence_text, row.career_text)
            for row in pool_texts.itertuples(index=False)
        ]
    else:
        late_rows = []
    if late_rows:
        late_df = pd.DataFrame(late_rows)
        pool_features = pd.concat([pool_features.reset_index(drop=True), late_df.reset_index(drop=True)], axis=1)
    else:
        for col in [
            "colbert_maxsim_score",
            "colbert_retrieval_score",
            "colbert_ranking_score",
            "colbert_eval_score",
            "colbert_production_ml_score",
            "colbert_python_score",
            "colbert_available",
        ]:
            pool_features[col] = 0.0

    raw_scores = predict_scores(pool_features, artifacts_dir)
    final_scores = apply_final_gates(pool_features, raw_scores)
    pool_features["final_score"] = final_scores
    ranked = select_top_candidates(pool_features, args.top_n)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked["score"] = calibrated_output_scores(ranked["final_score"].to_numpy())
    ranked["reasoning"] = [reasoning_for_row(row, int(row["rank"])) for row in ranked.to_dict("records")]

    out = Path(args.out)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        for row in ranked[["candidate_id", "rank", "score", "reasoning"]].to_dict("records"):
            writer.writerow(
                {
                    "candidate_id": row["candidate_id"],
                    "rank": int(row["rank"]),
                    "score": f"{float(row['score']):.6f}",
                    "reasoning": row["reasoning"],
                }
            )

    meta = {
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "pool_size": int(len(pool_features)),
        "output": str(out),
        "top_score_raw": float(ranked["final_score"].iloc[0]) if len(ranked) else None,
        "colbert_mode": "lexical_fallback",
        "dense_retrieval": "faiss_bge" if neural_index is not None else "svd_fallback",
    }
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()








