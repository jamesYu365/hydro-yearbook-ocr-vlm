# Data Spec

## Scope

This document defines the v0 data contract for the fixed-layout flow-table benchmark.

The v0 task contract is:
- input: single table image
- output: raw CSV text

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
- `layout_json_path`
- `perturbations`
- `source_template_id`
- `split`
- `source`

`source` is `synthetic` for training and validation data.

## Real Test Dataset

Real flow tables are test-only in v0.

The current real test manifest stores:
- `sample_id`
- `station_name`
- `river`
- `year`
- `pdf_path`
- `csv_path`
- `csv_encoding`
- `target_csv`
- `split`
- `source`

`source` is `real` and `split` is always `test`.

The current manifest is label-based. A later extraction step will add the real station-level table images used for final inference.

## CSV Rules

- The target is raw CSV text, not Markdown.
- Empty rows are preserved.
- Empty cells are preserved.
- Number formats are preserved exactly as generated or labeled.
- Evaluation uses raw model output without normalization, repair, or structure fixing.

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
