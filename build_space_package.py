"""Build a minimal, self-contained Hugging Face Space repository."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


FILES = {
    "app.py": "app.py",
    "demo_ranker.py": "demo_ranker.py",
    "redrob_ranker/__init__.py": "redrob_ranker/__init__.py",
    "redrob_ranker/features.py": "redrob_ranker/features.py",
    "redrob_ranker/late_interaction.py": "redrob_ranker/late_interaction.py",
    "sample_candidates.json": "India_runs_data_and_ai_challenge/sample_candidates.json",
}
REQUIREMENTS = """gradio==6.19.0\nnumpy==2.4.6\n"""
SPACE_README = """---
title: Bug Solvers Candidate Ranker
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 6.19.0
python_version: 3.11
app_file: app.py
pinned: false
short_description: CPU-only Redrob candidate ranking sandbox
tags:
  - ranking
  - recruiting
  - information-retrieval
---

# Bug Solvers Candidate Ranker

Hosted CPU-only sandbox for the Redrob Intelligent Candidate Discovery and Ranking Challenge.
It accepts a JSON or JSONL file containing at most 100 candidates and returns a deterministic ranked CSV.
No hosted API, external model, private candidate dataset, or secret is used at inference time.

Source: https://github.com/POPPz07/redrob_ai_challenge
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/huggingface_space")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = set(FILES) | {"README.md", "requirements.txt", "manifest.json"}
    unexpected = {
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file() and path.relative_to(output_dir).as_posix() not in expected
    }
    if unexpected:
        raise ValueError(f"Refusing to overwrite package with unexpected files: {sorted(unexpected)}")

    for destination, source in FILES.items():
        target = output_dir / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    (output_dir / "README.md").write_text(SPACE_README, encoding="utf-8")
    (output_dir / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")

    manifest = {
        "files": {
            path.relative_to(output_dir).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(output_dir.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        }
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir.resolve()), **manifest}, indent=2))


if __name__ == "__main__":
    main()