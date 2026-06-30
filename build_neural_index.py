from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from redrob_ranker.features import ASPECT_QUERIES
from redrob_ranker.neural_retrieval import MODEL_PATH, build_neural_artifacts


DEFAULT_QUERY = " ".join(ASPECT_QUERIES.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build resumable BGE + FAISS retrieval artifacts.")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--skip-retrain", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    artifacts_dir = Path(args.artifacts_dir)
    texts = pd.read_parquet(artifacts_dir / "candidate_text_views.parquet")
    metadata = build_neural_artifacts(
        texts,
        artifacts_dir,
        DEFAULT_QUERY,
        model_path=Path(args.model_path),
        batch_size=args.batch_size,
    )
    metadata["neural_precompute_seconds"] = round(time.perf_counter() - started, 3)
    metadata_path = artifacts_dir / "precompute_metadata.json"
    existing = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    existing.update(metadata)
    metadata_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    if not args.skip_retrain:
        subprocess.run(
            [
                sys.executable,
                "retrain_ranker.py",
                "--artifacts-dir",
                str(artifacts_dir),
            ],
            check=True,
        )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

