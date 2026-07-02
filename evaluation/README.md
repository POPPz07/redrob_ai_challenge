# Blind Human Evaluation

This directory provides a private, reproducible audit of ranking quality against human judgments. Generated candidate evidence, reviewer workbooks, ID mappings, labels, and reports belong under `outputs/` and must never be committed.

## Sample Design

The deterministic sample contains 200 unique profiles:

- 29: complete census of semantically verified rare-expert narratives.
- 31: high consistency-risk or honeypot cases.
- 45: strong non-rare profiles.
- 35: hard retrieval/ranking near-matches.
- 30: adjacent technical profiles.
- 20: moderate random profiles.
- 10: low-relevance controls.

The sample is deliberately enriched for hard cases. Its metrics diagnose this ranker; they are not unbiased estimates for all 100,000 candidates.

A private stratified split assigns 120 profiles to development and 80 to holdout. Reviewers cannot see the split, candidate IDs, model scores, current ranks, rarity bands, risk features, or sample strata.

## Generate The Pack

Activate the existing project venv and install the evaluation-only dependency inside it:

```powershell
& .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-evaluation.txt
python evaluation\create_annotation_sample.py
```

Build the two independent `.xlsx` workbooks with the configured Codex spreadsheet runtime, then finalize them:

```powershell
node evaluation\build_annotation_workbooks.mjs
python evaluation\finalize_annotation_workbooks.py
```

The current verified files are:

```text
outputs/annotation_audit/annotation_reviewer_1.xlsx
outputs/annotation_audit/annotation_reviewer_2.xlsx
outputs/annotation_audit/annotation_key.csv
```

## Reviewer Procedure

Use two reviewers who understand applied ML and information retrieval. Each person receives only their own workbook.

1. Read `Instructions` and `Rubric` first.
2. Label every profile independently without discussing candidates.
3. Enter relevance `0-5`, confidence `1-3`, risk flag `No/Unsure/Yes`, and a rationale of at least 15 characters.
4. Use only visible profile facts. Do not infer identity or protected characteristics.
5. Return the workbook without changing profile keys or sheet names.

## Evaluate

After both reviewers finish, calculate agreement and development metrics only:

```powershell
python evaluation\evaluate_annotations.py --require-complete
```

After all development decisions are frozen, unlock the one-time holdout report:

```powershell
python evaluation\evaluate_annotations.py --require-complete --unlock-holdout
```

Outputs:

```text
outputs/annotation_audit/evaluation/human_evaluation_report.json
outputs/annotation_audit/evaluation/stratum_metrics.csv
outputs/annotation_audit/evaluation/adjudication_queue_private.csv
```

The report includes exact agreement, within-one agreement, quadratic weighted Cohen's kappa, risk agreement, NDCG at 10/20/50/100, mean relevance, precision for relevance 4+, Spearman correlation, top-100 recall, and separate development/holdout metrics.

Metric implementations follow the official [scikit-learn NDCG](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html), [weighted Cohen's kappa](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html), and [SciPy Spearman](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html) APIs.

## Decision Protocol

- Resolve cases with relevance disagreement greater than one point or different risk flags.
- If quadratic weighted kappa is below `0.70` or within-one agreement is below `0.85`, calibrate reviewers and relabel before tuning.
- Use only development labels for model/feature/threshold decisions; the evaluator hides holdout metrics by default.
- Freeze one candidate configuration before opening holdout metrics.
- Promote a change only if holdout NDCG@10 and NDCG@50 improve without validator, runtime, memory, determinism, or safety regressions.
- Never present synthetic smoke-test labels or the local narrative proxy as human ground truth.