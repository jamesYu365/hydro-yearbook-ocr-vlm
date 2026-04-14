# Real Test Extraction

## Goal

Extract station-level table images from the source yearbook PDFs for final-test inference.

The current extraction step is:
- render each PDF page to a page image
- run layout detection on the page image
- save a plain crop from the detected table box
- save a buffered crop with extra space added above the table
- build a title ROI from the top buffer strip plus a few pixels below the original table top
- run title OCR only on that title ROI
- rename the three saved artifacts with the same final basename

The current extraction step does not:
- perform full-table OCR
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
pip install rapidocr_onnxruntime
```

Then verify:

```bash
conda run -n rapid python -c "import fitz; print('fitz ok')"
conda run -n rapid python -c "import cv2; print('cv2 ok')"
conda run -n rapid python -c "from rapid_table_det.inference import TableDetector; print('rapid_table_det ok')"
conda run -n rapid python -c "from rapidocr_onnxruntime import RapidOCR; print('rapidocr ok')"
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
- `station_tables_plain/`: plain table crops
- `station_tables_buffered/`: buffered table crops
- `title_rois/`: OCR input images for title recognition
- `layout_detections.json`: page-level detection metadata
- `layout_failures.jsonl`: per-page failures

The run-level summary is written to:
- `datasets/derived/final_test_layout/run_summary.json`

## Usage

Process the current default target PDFs:

```bash
conda run -n rapid python datasets/extract_station_tables.py
```

Use a larger or smaller title buffer during tuning:

```bash
conda run -n rapid python datasets/extract_station_tables.py --title-buffer-px 220
```

Tune the small downward compensation under the title ROI:

```bash
conda run -n rapid python datasets/extract_station_tables.py --title-bottom-buffer-px 16
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

The baseline extraction run completed successfully on 2026-04-13 with:
- `流量 2006`: `18` pages and `35` detected station tables
- `水位 2014`: `24` pages and `47` detected station tables
- layout detection failure count: `0` for both PDFs

Title OCR success is stricter than layout success:
- `流量` titles only count as success if OCR text contains `逐日平均流量表`
- `水位` titles only count as success if OCR text contains `逐日平均水位表`
- final filenames use the form `稳定ID_标题_年份`
- `标题` is extracted from the OCR text before `逐日平均流量表` or `逐日平均水位表`
- the leading ordinal digit is removed from that extracted prefix
- if the keyword is missing, adjust `--title-buffer-px` first and rerun a small page subset

## Next Step

After extraction, the next dataset task is:
- align cropped station-table images with the calibrated CSV labels
- build the real final-test inference manifest from those aligned pairs
