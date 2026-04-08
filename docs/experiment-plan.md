# Experiment Plan

## v0 Goal

Run a single-model baseline with `GOT-OCR2.0` on synthetic flow tables and evaluate on the real calibrated flow test set.

## Fixed Decisions

- task: single table image to CSV
- domain: flow tables only
- layout scope: one fixed flow-table layout
- training source: synthetic only
- validation source: synthetic holdout
- test source: real flow tables only
- model output: raw CSV with no post-processing
- preferred training path: `ms-swift + LoRA`

## Deliverables

- a reproducible synthetic dataset manifest
- a real test manifest
- a real table-image extraction step for final inference
- a `ms-swift` training manifest and wrapper for `GOT-OCR2.0`
- a strict evaluation report
- a per-station score table
- representative failure cases

## Metrics

- weighted score: 30% character accuracy + 70% cell accuracy
- reference metrics:
  - character accuracy
  - cell accuracy
- error tags:
  - truncation
  - structure_error
  - value_error

## Current Execution Order

1. validate the Linux `rapid` environment
2. run a small synthetic smoke test
3. build train and validation `ms-swift` manifests
4. extract real station-level table images from the PDF
5. launch the first `GOT-OCR2.0` LoRA run
6. evaluate on the real test set without output repair
