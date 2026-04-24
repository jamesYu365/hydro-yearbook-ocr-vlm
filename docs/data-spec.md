# Data Spec

## Scope

This document defines the v0 data contract for the fixed-layout flow-table benchmark.

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

## Real Test Dataset

Real flow tables are test-only in v0.

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

The current official real-test image set for `2006 流量` is `station_tables_daily/`, which is produced by the OCR-based daily-only crop path.
The same OCR-based daily-only crop path is also validated for `2014 水位`, but no aligned manifest exists for it yet because calibrated CSV labels are not present in the repository.

The current aligned manifest is produced by:
- `datasets/crop_flow_table_daily_region.py`
- `datasets/build_real_flow_alignment.py`

## Output Rules

- `target_csv` preserves the calibrated table label exactly.
- `target_got_format` is deterministically derived from `target_csv`.
- Empty rows and empty cells are preserved when deriving either representation.
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
