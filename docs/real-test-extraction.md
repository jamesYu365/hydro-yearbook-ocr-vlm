# Real Test Extraction

## Goal

Extract station-level table images from the source yearbook PDFs for final-test inference.

The current extraction step is:
- render each PDF page to a page image
- run layout detection on the page image
- crop each detected table region into a station-level table image

The current extraction step does not:
- perform OCR
- split a station table into header/day/month/year sub-images
- build the final inference manifest automatically

## Current Target PDFs

The current validated inputs are:
- `datasets/流量/2006 流量 6-16 汉江区（汉江下游水系，汈汊湖、东荆河水系）.pdf`
- `datasets/水位/2014 水位 6-16 汉江区（汉江下游水系、汈汊湖、东荆河水系）.pdf`

## Environment

Run this step in the `rapid` conda environment, not the `got` environment.

Reason:
- layout detection depends on `rapid_table_det`
- PDF rendering and crop writing were validated in `rapid`
- training dependencies and extraction dependencies should stay decoupled

Recommended environment bootstrap:

```bash
conda create -n rapid python=3.10 -y
conda activate rapid
pip install pymupdf opencv-python
pip install rapid_table_det
```

Then verify:

```bash
conda run -n rapid python -c "import fitz; print('fitz ok')"
conda run -n rapid python -c "import cv2; print('cv2 ok')"
conda run -n rapid python -c "from rapid_table_det.inference import TableDetector; print('rapid_table_det ok')"
```

## Script Location And Code Rules

The extraction script lives under:
- `datasets/extract_station_tables.py`

Keep this type of code under `datasets/` because it is data-preparation logic tied directly to the source PDFs and derived test assets.

Keep these boundaries:
- source PDFs and calibrated CSV labels stay under `datasets/`
- derived page images and cropped station tables stay under `datasets/derived/`
- model-specific preprocessing stays under `scripts/models/`
- synthetic-data generation stays under `scripts/data/`

For this extraction path:
- do not modify or rename the original PDF files
- do not write derived images next to the source-of-truth CSV labels
- do not add station-level secondary splits in this script unless a later workflow genuinely needs them
- keep outputs reproducible from the original PDFs plus the extraction script

## Output Layout

The current script writes outputs under:

```text
datasets/derived/final_test_layout/
```

For each PDF it creates:
- `pages/`: rendered page images
- `station_tables/`: cropped station-level table images
- `layout_detections.json`: page-level detection metadata
- `layout_failures.jsonl`: per-page failures

The run-level summary is written to:
- `datasets/derived/final_test_layout/run_summary.json`

## Usage

Process the current default target PDFs:

```bash
conda run -n rapid python datasets/extract_station_tables.py
```

Smoke-test only the first page of each PDF:

```bash
conda run -n rapid python datasets/extract_station_tables.py --limit-pages 1
```

Reuse rendered pages if they already exist:

```bash
conda run -n rapid python datasets/extract_station_tables.py --skip-rendered-pages
```

Reuse existing table crops for pages that were already extracted:

```bash
conda run -n rapid python datasets/extract_station_tables.py --skip-existing-crops
```

Process a custom PDF:

```bash
conda run -n rapid python datasets/extract_station_tables.py \
  --pdf "datasets/流量/2006 流量 6-16 汉江区（汉江下游水系，汈汊湖、东荆河水系）.pdf"
```

## Current Validated Result

The extraction run completed successfully on 2026-04-13 with:
- `流量 2006`: `18` pages and `35` cropped station tables
- `水位 2014`: `24` pages and `47` cropped station tables
- failure count: `0` for both PDFs

## Next Step

After extraction, the next dataset task is:
- align cropped station-table images with the calibrated CSV labels
- build the real final-test inference manifest from those aligned pairs
