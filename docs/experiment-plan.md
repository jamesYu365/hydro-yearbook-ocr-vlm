# Experiment Plan

## Goal

Run a single-model baseline with `GOT-OCR2.0` on synthetic flow tables and evaluate on the real calibrated flow test set.

## Fixed Decisions

- task: single table image to official GOT formatted table text
- domain: flow tables only
- layout scope: one fixed flow-table layout
- training source: synthetic only
- validation source: synthetic holdout
- test source: real flow tables only
- label authority: calibrated CSV
- model output for the current baseline: official GOT formatted table text
- preferred training path: `ms-swift + LoRA`

## Deliverables

- a reproducible synthetic dataset manifest
- a real test manifest
- a real table-image extraction step for final inference
- a `ms-swift` training manifest and wrapper for `GOT-OCR2.0`
- a formal evaluation report aligned with the chosen target format
- a per-station score table
- representative failure cases

## Current Verified Progress

- unit tests are passing in `got`
- synthetic data smoke generation has passed
- the real test manifest has been built with 35 records
- v1 `train_swift.jsonl` and `val_swift.jsonl` have been built with `8000` train records and `2000` validation records
- a 1-step GOT-OCR2.0 LoRA smoke run has passed with `sdpa`
- the v1 targets remove empty artifact rows and trailing empty artifact columns while preserving empty cells inside data rows

## Metrics

- primary metrics should match the active target format
- current GOT runs produce raw or pretty LaTeX table text, so evaluation should use the shared table parser
- character accuracy is computed on parsed table cell text when parsing succeeds, not on raw LaTeX markup

## Current Execution Order

1. extract real station-level table images from the PDF
2. train `GOT-OCR2.0` LoRA from the v1 synthetic Swift manifests
3. rerun base and fine-tuned inference on the real extracted test set
4. run table-format evaluation and model comparison on raw/pretty GOT outputs
5. inspect representative failure cases and tag them
