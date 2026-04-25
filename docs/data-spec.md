# Data Spec

## Scope

This document defines the current data contract for the fixed-layout flow-table benchmark.

The repository currently tracks two related output contracts:
- label contract: calibrated CSV remains the source-of-truth representation
- current GOT baseline contract: official GOT formatted table text

## Synthetic Dataset

Synthetic data is stored as:
- rendered table images
- one layout JSON per image
- `train.jsonl` and `val.jsonl` manifests

The generator lives under `scripts/data/`. Model-specific conversion code lives under `scripts/models/`.

Each synthetic manifest row must include:
- `sample_id`
- `image_path`
- `target_csv`
- `target_got_format`
- `layout_json_path`
- `perturbations`
- `source_template_id`
- `split`
- `source`

`source` is `synthetic` for training and validation data.

The current v1 synthetic manifest split is a strict seeded shuffle split:
- generation command: `--num-samples 10000 --val-ratio 0.2 --seed 20260408 --num-workers 16`
- train records: `8000`
- validation records: `2000`

## Real Test Dataset

Real flow tables are test-only in the current benchmark.

The current aligned real test manifest stores:
- `sample_id`
- `station_name`
- `river`
- `year`
- `csv_path`
- `csv_encoding`
- `image_path`
- `title_text`
- `target_csv`
- `target_got_format`
- `match_score`
- `match_status`
- `split`
- `source`

`source` is `real` and `split` is always `test`.
The current aligned real flow test manifest contains `35` records.

The current official real-test image set for `2006 流量` is `station_tables_daily/`, which is produced by the OCR-based daily-only crop path.
The same OCR-based daily-only crop path is also validated for `2014 水位`, but no aligned manifest exists for it yet because calibrated CSV labels are not present in the repository.

The current aligned manifest is produced by:
- `datasets/crop_flow_table_daily_region.py`
- `datasets/build_real_flow_alignment.py`

## Output Rules

- Raw calibrated CSV files remain the source-of-truth labels and are referenced by `csv_path` in real-test manifests.
- Manifest `target_csv` is derived from the label rows after removing only fully empty separator rows.
- Manifest `target_got_format` is deterministically derived from the same cleaned target rows.
- Empty cells inside otherwise non-empty rows are preserved.
- Natural calendar blanks and labeled missing values are preserved.
- Fully empty separator rows are omitted from manifest targets, even when they remain visible in rendered images.
- Number formats are preserved exactly as generated or labeled.
- The current `ms-swift` GOT training path defaults to:
  - prompt: `OCR with format: `
  - response field: `target_got_format`
- Strict CSV evaluation remains available, but it should only be treated as a like-for-like metric when the prediction target is also CSV.

## Layout JSON

Each layout JSON must include:
- `sample_id`
- `image_path`
- `table_meta`
- `cells`
- `perturbations`

Each cell record should include:
- `row_index`
- `col_index`
- `text`
- `bbox`
- `cell_type`

`cell_type` is typically `header`, `index`, `data`, or `separator`.

## Label Authority

- Calibrated CSV files under `datasets/流量/<year>/` are the source-of-truth labels.
- Existing dataset filenames should be preserved.
- Any derived manifest or rendered sample must trace back to those filenames and labels.
