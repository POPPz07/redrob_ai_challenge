from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Sequence

import numpy as np


MODEL_NAME = "BAAI/bge-small-en-v1.5"
MODEL_PATH = Path("models/bge-small-en-v1.5")
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
MAX_SEQUENCE_LENGTH = 128


def compressed_documents(texts) -> list[str]:
    evidence = texts["evidence_text"].fillna("").astype(str)
    career = texts["career_text"].fillna("").astype(str).str.slice(0, 500)
    return (evidence + " " + career).str.strip().tolist()


def load_local_model(model_path: Path):
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(min(12, os.cpu_count() or 4))
    model = SentenceTransformer(str(model_path), device="cpu", local_files_only=True)
    model.max_seq_length = MAX_SEQUENCE_LENGTH
    return model


def encode_documents_resumable(
    documents: Sequence[str],
    artifacts_dir: Path,
    model_path: Path,
    batch_size: int = 128,
    checkpoint_rows: int = 1024,
) -> Path:
    embedding_path = artifacts_dir / "neural_embeddings.tmp.npy"
    progress_path = artifacts_dir / "neural_embeddings.progress.json"
    total = len(documents)
    model = load_local_model(model_path)
    dimension = int(model.get_embedding_dimension())

    completed = 0
    mode = "w+"
    if embedding_path.exists() and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("total") == total and progress.get("dimension") == dimension:
            completed = int(progress.get("completed", 0))
            mode = "r+"
    embeddings = np.lib.format.open_memmap(
        embedding_path, mode=mode, dtype=np.float32, shape=(total, dimension)
    )

    started = time.perf_counter()
    for start in range(completed, total, checkpoint_rows):
        end = min(start + checkpoint_rows, total)
        embeddings[start:end] = model.encode_document(
            list(documents[start:end]),
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        embeddings.flush()
        progress_path.write_text(
            json.dumps({"total": total, "dimension": dimension, "completed": end}, indent=2),
            encoding="utf-8",
        )
        elapsed = time.perf_counter() - started
        rate = (end - completed) / max(elapsed, 0.001)
        remaining_minutes = (total - end) / max(rate, 0.001) / 60.0
        print(
            f"neural embeddings {end:,}/{total:,}; "
            f"{rate:.1f} docs/s; ~{remaining_minutes:.1f} min remaining",
            flush=True,
        )
    return embedding_path


def build_faiss_index(embedding_path: Path, artifacts_dir: Path) -> dict[str, float | int | str]:
    import faiss

    embeddings = np.load(embedding_path, mmap_mode="r")
    count, dimension = embeddings.shape
    index = faiss.IndexScalarQuantizer(
        dimension,
        faiss.ScalarQuantizer.QT_8bit,
        faiss.METRIC_INNER_PRODUCT,
    )
    sample_size = min(20000, count)
    sample_ids = np.linspace(0, count - 1, sample_size, dtype=np.int64)
    index.train(np.asarray(embeddings[sample_ids], dtype=np.float32))
    for start in range(0, count, 10000):
        index.add(np.asarray(embeddings[start : start + 10000], dtype=np.float32))
    index_path = artifacts_dir / "dense_index.faiss"
    faiss.write_index(index, str(index_path))
    return {
        "faiss_index_type": "IndexScalarQuantizer(QT_8bit, inner_product)",
        "faiss_count": int(index.ntotal),
        "faiss_dimension": int(dimension),
        "faiss_index_bytes": int(index_path.stat().st_size),
    }


def encode_query(query: str, model_path: Path, artifacts_dir: Path) -> None:
    model = load_local_model(model_path)
    vector = model.encode_query(
        [QUERY_INSTRUCTION + query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)
    np.save(artifacts_dir / "dense_query_embedding.npy", vector)


def build_neural_artifacts(
    texts,
    artifacts_dir: Path,
    query: str,
    model_path: Path = MODEL_PATH,
    batch_size: int = 128,
) -> dict[str, float | int | str | bool]:
    if not model_path.exists():
        return {
            "faiss_available": False,
            "neural_model": str(model_path),
            "neural_error": "local model directory not found",
        }
    documents = compressed_documents(texts)
    embedding_path = encode_documents_resumable(
        documents, artifacts_dir, model_path, batch_size=batch_size
    )
    metadata = build_faiss_index(embedding_path, artifacts_dir)
    encode_query(query, model_path, artifacts_dir)
    progress_path = artifacts_dir / "neural_embeddings.progress.json"
    embedding_path.unlink(missing_ok=True)
    progress_path.unlink(missing_ok=True)
    metadata.update(
        {
            "faiss_available": True,
            "neural_model": MODEL_NAME,
            "neural_model_path": str(model_path),
            "neural_max_sequence_length": MAX_SEQUENCE_LENGTH,
        }
    )
    return metadata
