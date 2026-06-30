from __future__ import annotations

import re
from collections import Counter

from .features import ASPECT_QUERIES, clean_text


USE_COLBERT = True


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9+#.]+", clean_text(text).lower())


QUERY_TOKENS = {name: tokenize(query) for name, query in ASPECT_QUERIES.items()}


def lexical_maxsim(
    query_tokens: list[str],
    document_counts: Counter[str],
    document_prefixes: set[str],
) -> float:
    """CPU-safe lexical approximation of contextual-token MaxSim."""
    if not query_tokens or not document_counts:
        return 0.0
    score = 0.0
    for token in query_tokens:
        if token in document_counts:
            score += min(1.0, 0.65 + 0.12 * document_counts[token])
        elif len(token) >= 5 and token[:5] in document_prefixes:
            score += 0.35
    return score / len(query_tokens)


def score_late_interaction(evidence_text: str, career_text: str = "") -> dict[str, float]:
    doc = clean_text(f"{evidence_text} {career_text[:900]}")
    document_counts = Counter(tokenize(doc))
    document_prefixes = {token[:5] for token in document_counts if len(token) >= 5}
    scores = {
        name: lexical_maxsim(query_tokens, document_counts, document_prefixes)
        for name, query_tokens in QUERY_TOKENS.items()
    }
    retrieval = scores.get("retrieval", 0.0)
    ranking = scores.get("ranking", 0.0)
    evaluation = scores.get("evaluation", 0.0)
    production = scores.get("production_ml", 0.0)
    python = scores.get("python", 0.0)
    max_score = max(scores.values()) if scores else 0.0
    return {
        "colbert_maxsim_score": max_score,
        "colbert_retrieval_score": retrieval,
        "colbert_ranking_score": ranking,
        "colbert_eval_score": evaluation,
        "colbert_production_ml_score": production,
        "colbert_python_score": python,
        "colbert_available": 0.0,
    }
