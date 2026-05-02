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
- `data_regime`
- `source_template_id`
- `split`
- `source`

`source` is `synthetic` for training and validation data.

The current v1 synthetic manifest split is a strict seeded shuffle split:
- generation command: `--num-samples 10000 --val-ratio 0.2 --seed 20260408 --num-workers 16`
- train records: `8000`
- validation records: `2000`

For the v2 synthetic loop, use `--dataset-version flow_v2`, `--output-dir data/flow_v2`, and `--manifest-dir data/manifests/flow_v2`. The v2 manifest keeps the same row contract and uses `data_regime` to identify normal, zero-heavy, zero-with-spikes, and calendar-tail-focus samples.

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
- Manifest `target_csv` is derived from the label rows after removing empty artifact rows and trailing empty artifact columns.
- Manifest `target_got_format` is deterministically derived from the same cleaned target rows.
- Empty cells inside otherwise non-empty rows are preserved.
- Natural calendar blanks and labeled missing values are preserved.
- Empty separator rows and trailing empty CSV-export columns are omitted from manifest targets, even when they remain visible in rendered images or source CSV files.
- Number formats are preserved exactly as generated or labeled.
- The current `ms-swift` GOT training path defaults to:
  - prompt: `OCR with format: `
  - response field: `target_got_format`
- Current inference produces raw GOT or pretty LaTeX table text. The evaluator parses those prediction rows and compares
  them to the manifest `target_csv` source-of-truth cells.

## Layout JSON

Each layout JSON must include:
- `sample_id`
- `image_path`
- `table_meta`
- `cells`
- `perturbations`

For synthetic samples, `table_meta` also stores the `data_regime` used to generate the table.

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
