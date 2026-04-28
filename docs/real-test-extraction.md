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

The repository now also includes follow-up scripts for the extracted station-table crops:
- `datasets/crop_flow_table_daily_region.py`: crop existing `station_tables_plain/` images down to the daily-only region by scanning a lower-left statistics ROI, then anchoring on full `平均` or constrained `平/均` weak anchors, writing to `station_tables_daily/`
- `datasets/build_real_flow_alignment.py`: score and align calibrated CSV labels to OCR-cropped JPG files and emit a real-test manifest plus an audit report
- `datasets/run_ocr_on_image.py`: run RapidOCR on any single debug image or ROI and print token-level OCR output

These dataset-side scripts are thin entrypoints now. Shared implementation lives under:
- `src/yearbook_ocr/data/`
- `src/yearbook_ocr/ocr/`

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
pip install rapidocr==3.8.1
```

Then verify:

```bash
conda run -n rapid python -c "import fitz; print('fitz ok')"
conda run -n rapid python -c "import cv2; print('cv2 ok')"
conda run -n rapid python -c "from rapid_table_det.inference import TableDetector; print('rapid_table_det ok')"
conda run -n rapid python -c "from rapidocr import RapidOCR; print('rapidocr ok')"
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
- align cropped station-table images with the calibrated CSV labels when those labels exist
- build the real final-test inference manifest from those aligned pairs

For the aligned real-test manifest, `csv_path` points to the original calibrated CSV file. The manifest `target_csv` and `target_got_format` are derived from that CSV after removing empty artifact rows and trailing empty artifact columns; crop generation and source CSV files are unchanged.

Current helper commands for that step:

```bash
conda run -n rapid python datasets/crop_flow_table_daily_region.py
conda run -n rapid python datasets/run_ocr_on_image.py 'datasets/derived/debug_rois/2014_水位/page_0004_table1_汉江汉川站_2014.jpg' --ocr-cuda off
python3 datasets/build_real_flow_alignment.py
```

## Daily Crop Notes

The OCR-based daily crop path is now validated for both current source PDFs:
- `2006 流量`
- `2014 水位`

The current daily-crop behavior is:
- build a lower-left statistics ROI from the extracted plain crop
- run OCR only on that ROI, not on the full table image
- default to `rapidocr` with the Paddle backend, `PP-OCRv5` server models, and physical `GPU 1` exposed as Paddle `gpu:0`
- choose the upper monthly-statistics anchor from OCR tokens
- prefer a full `平均` token
- if full `平均` is missing, allow `平` or `均` as constrained weak anchors only when they appear in the left statistics label column and remain clearly above `年统计`
- never use `年统计` itself as the formal crop anchor
- add a small bottom buffer so the `31` row is not clipped

The current validated `2014 水位` result is:
- `47` input tables
- `47` daily crops
- `0` failures
- `28` crops with `average_anchor`
- `19` crops with `weak_average_anchor`
