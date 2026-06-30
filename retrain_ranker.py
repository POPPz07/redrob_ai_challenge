from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import normalize
from xgboost import XGBRegressor

import rank
from redrob_ranker.features import NUMERIC_FEATURES, clamp, pseudo_label, stable_minmax


MODEL_FEATURES = list(NUMERIC_FEATURES) + [
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain weakly supervised XGBoost on real retrieval features.")
    parser.add_argument("--artifacts-dir", default="artifacts")
    args = parser.parse_args()
    artifacts = Path(args.artifacts_dir)
    features, _, vectorizer, tfidf, svd, svd_embeddings, neural_index, neural_query = (
        rank.load_artifacts(artifacts)
    )

    query_vector = vectorizer.transform([rank.DEFAULT_QUERY])
    sparse_scores = (tfidf @ query_vector.T).toarray().ravel()
    query_svd = normalize(svd.transform(query_vector), norm="l2", copy=False).astype(np.float32)
    svd_scores = np.asarray(svd_embeddings @ query_svd.T).ravel()
    dense_scores = svd_scores.copy()
    if neural_index is not None and neural_query is not None:
        distances, indices = neural_index.search(neural_query, neural_index.ntotal)
        dense_scores = np.zeros(len(features), dtype=np.float32)
        dense_scores[indices[0]] = distances[0]

    rule_scores = (
        0.28 * features["career_system_fit"].to_numpy()
        + 0.22 * features["retrieval_ranking_depth"].to_numpy()
        + 0.16 * features["production_ml_depth"].to_numpy()
        + 0.12 * features["evaluation_experimentation_fit"].to_numpy()
        + 0.08 * features["high_signal_title"].to_numpy()
        + 0.08 * features["skill_trust_score"].to_numpy()
        + 0.06 * features["weighted_formula_score"].to_numpy()
        - 0.20 * features["honeypot_risk"].to_numpy()
        - 0.12 * features["keyword_stuffing_penalty"].to_numpy()
    )
    train = features.copy()
    train["sparse_score"] = sparse_scores
    train["dense_score"] = dense_scores
    train["rule_recall_score"] = rule_scores
    for column in MODEL_FEATURES:
        if column not in train:
            train[column] = 0.0

    base_labels = np.array(
        [pseudo_label(row) for row in train.to_dict("records")], dtype=np.float32
    )
    sparse_norm = np.asarray(stable_minmax(sparse_scores.tolist()), dtype=np.float32)
    dense_norm = np.asarray(stable_minmax(dense_scores.tolist()), dtype=np.float32)
    labels = np.array(
        [
            clamp(0.88 * base + 0.05 * sparse_value + 0.07 * dense_value)
            for base, sparse_value, dense_value in zip(base_labels, sparse_norm, dense_norm)
        ],
        dtype=np.float32,
    )
    matrix = train[MODEL_FEATURES].astype(np.float32).to_numpy()
    model = XGBRegressor(
        n_estimators=220,
        max_depth=4,
        learning_rate=0.035,
        subsample=0.88,
        colsample_bytree=0.86,
        objective="reg:squarederror",
        n_jobs=4,
        random_state=42,
        tree_method="hist",
    )
    model.fit(matrix, labels)
    payload = {
        "model": model,
        "feature_cols": MODEL_FEATURES,
        "training": {
            "label_type": "weak_supervision_with_real_retrieval",
            "dense_source": "faiss_bge" if neural_index is not None else "svd_fallback",
            "candidate_count": len(features),
        },
    }
    joblib.dump(payload, artifacts / "ranker_model.joblib")
    importance = sorted(
        ((name, float(value)) for name, value in zip(MODEL_FEATURES, model.feature_importances_)),
        key=lambda item: item[1],
        reverse=True,
    )[:15]
    report = {"training": payload["training"], "top_feature_importance": importance}
    (artifacts / "ranker_training_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

