# Synthetic Data

## Objective

Generate high-fidelity synthetic flow tables that match the fixed real test layout closely enough to validate the end-to-end OCR pipeline.

The synthetic dataset is the primary training source for the current flow-table benchmark.

## Rendering Rules

- Render the full table programmatically.
- Do not reuse real table screenshots.
- Match table skeleton first: row heights, column widths, borders, header structure, alignment.
- Match font system next: font family, size, weight, and numeric appearance.
- Use a Chinese-capable font for all rendered text. Do not use Latin-only defaults such as DejaVu Sans for the final dataset.
- Match line width, margins, and whitespace after that.
- Add print and scan texture last.

The current default render configuration is:
- image size: `2160x1090`
- `margin_x=60`
- `margin_y=35`
- `header_height=70`
- `row_height=25`
- `body_font_size=17`

The top-left header cell is rendered visually as a diagonal day/month header. The manifest target still stores the normalized text label as `日\月`.

## Content Rules

- Non-data text stays fixed to the template.
- Numeric values are generated independently and must not copy a real test table verbatim.
- Value generation should follow the observed formatting distribution in the real yearbook data.
- Natural calendar blanks are preserved.
- Randomly blanking valid dates is not part of the current data contract.
- A small amount of missing values in data cells is allowed.
- Blank separator rows remain in the rendered image and layout JSON.
- Manifest targets remove empty artifact rows and trailing empty artifact columns.
- Empty cells inside non-empty data rows remain in both `target_csv` and `target_got_format`.

## Perturbation Rules

- Geometric and lighting perturbations are primary.
- Content perturbations are secondary.
- The target mix is:
  - 10% clean
  - 45% single perturbation
  - 35% double perturbation
  - 10% three or more perturbations

The current generator includes:
- brightness
- contrast
- blur
- shadow
- light geometric transform

## Reproducibility

Synthetic generation should remain reproducible through:
- a checked-in config
- a fixed seed
- a versioned output directory
- per-sample perturbation metadata
- layout JSON for each rendered sample

## Current Implementation Notes

The current generator:
- discovers a template from calibrated flow CSV files
- reuses the fixed table structure
- builds month-wise numeric pools from real labels
- samples new numeric values independently
- auto-selects a system Chinese font when `--font-path` is not provided
- writes image, layout JSON, and manifest outputs together

Current recommended generation command:

```bash
python3 ./scripts/data/generate_synthetic_flow_v0.py --num-samples 10000 --val-ratio 0.2 --seed 20260408 --num-workers 16
```

The current generated manifest sizes are:
- `data/manifests/flow_v0/train.jsonl`: `8000`
- `data/manifests/flow_v0/val.jsonl`: `2000`

Operational notes:
- `--num-workers` parallelizes image and layout generation.
- `--manifest-only` rebuilds `train.jsonl` and `val.jsonl` from existing images and layouts without rewriting those assets.
- the train/validation split is a strict seeded shuffle split, not a contiguous index split.

## v2 Direction

The next synthetic dataset version targets the current real-test failure modes:
- blank-vs-literal-`0` confusion
- zero-heavy station tables
- calendar-tail rows `29`, `30`, and `31`

Recommended v2 generation command:

```bash
conda run --no-capture-output -n got python scripts/data/generate_synthetic_flow_v0.py \
  --dataset-dir datasets/流量/2006 \
  --output-dir data/flow_v2 \
  --manifest-dir data/manifests/flow_v2 \
  --dataset-version flow_v2 \
  --num-samples 8000 \
  --val-ratio 0.2 \
  --seed 20260428 \
  --num-workers 16
```

The v2 generator records `data_regime` in each manifest row and layout JSON. The current regime mix is:
- `normal`: 45%
- `zero_heavy`: 25%
- `zero_with_spikes`: 15%
- `calendar_tail_focus`: 15%

`flow_v0` generation keeps the legacy normal sampling profile for reproducibility. The hard-regime mix is activated by using a new dataset version such as `flow_v2`.
