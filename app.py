from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr

from demo_ranker import rank_demo_file


def rank_uploaded_candidates(uploaded_path: str) -> str:
    if not uploaded_path:
        raise gr.Error("Upload a JSON or JSONL candidate file.")
    output_dir = Path(tempfile.mkdtemp(prefix="redrob_demo_"))
    try:
        return str(rank_demo_file(uploaded_path, output_dir / "ranked_candidates.csv"))
    except (ValueError, KeyError, TypeError) as exc:
        raise gr.Error(str(exc)) from exc


demo = gr.Interface(
    fn=rank_uploaded_candidates,
    inputs=gr.File(
        label="Candidate JSON or JSONL",
        file_types=[".json", ".jsonl"],
        type="filepath",
    ),
    outputs=gr.File(label="Ranked CSV"),
    title="Bug Solvers Candidate Ranker",
    description="CPU-only sample ranking for up to 100 Redrob candidate profiles.",
)


if __name__ == "__main__":
    demo.launch()
