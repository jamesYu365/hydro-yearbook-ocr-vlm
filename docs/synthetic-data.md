# Synthetic Data

## v0 Objective

Generate high-fidelity synthetic flow tables that match the fixed real test layout closely enough to validate the end-to-end OCR pipeline.

The synthetic dataset is the primary training source for v0.

## Rendering Rules

- Render the full table programmatically.
- Do not reuse real table screenshots.
- Match table skeleton first: row heights, column widths, borders, header structure, alignment.
- Match font system next: font family, size, weight, and numeric appearance.
- Use a Chinese-capable font for all rendered text. Do not use Latin-only defaults such as DejaVu Sans for the final dataset.
- Match line width, margins, and whitespace after that.
- Add print and scan texture last.

## Content Rules

- Non-data text stays fixed to the template.
- Numeric values are generated independently and must not copy a real test table verbatim.
- Value generation should follow the observed formatting distribution in the real yearbook data.
- Natural calendar blanks are preserved.
- Randomly blanking valid dates is not part of v0.
- A small amount of missing values in data cells is allowed.

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

The current v0 generator:
- discovers a template from calibrated flow CSV files
- reuses the fixed table structure
- builds month-wise numeric pools from real labels
- samples new numeric values independently
- auto-selects a system Chinese font when `--font-path` is not provided
- writes image, layout JSON, and manifest outputs together
