# Project Overview

## Summary

Hydro Yearbook OCR VLM is a focused project for fine-tuning a vision-language model on hydrological yearbook OCR, starting from fixed-layout flow tables.

The immediate objective is:
- fine-tune `GOT-OCR2.0`
- train mainly on high-fidelity synthetic flow tables
- evaluate on real calibrated flow tables
- keep calibrated CSV as label authority while the current GOT training target follows the official `OCR with format:` output style

## Current Repository State

The repository is no longer documentation-only. The current implemented pieces are:
- source PDFs and calibrated CSV labels under `datasets/`
- real-test PDF page rendering and layout-based station-table extraction under `datasets/`
- OCR-based daily-only crop generation for real `2006 流量` tables under `datasets/`
- scored alignment from extracted daily crops to calibrated `2006 流量` CSV labels under `datasets/`
- a shared core package under `src/yearbook_ocr/`
- synthetic flow-table generation under `scripts/data/`
- `ms-swift` manifest conversion for `GOT-OCR2.0` under `scripts/models/got_ocr2/`
- unified GOT inference through `scripts/models/got_ocr2/run_inference.py`
- raw GOT/LaTeX prediction evaluation through `scripts/eval/`
- a Linux environment setup script under `scripts/`
- basic tests under `tests/`
- a validated `swift sft` wrapper under `scripts/models/got_ocr2/`

Existing generated artifacts under `data/` already include:
- small synthetic sample images
- layout JSON files
- synthetic manifests
- a real test manifest
- a `ms-swift` manifest preview

The current verified status as of 2026-04-24 is:
- focused data, manifest, inference, and evaluation suites pass in `got`
- synthetic data generation smoke test has passed
- real test manifest generation has passed with 35 records
- v1 `train_swift.jsonl` and `val_swift.jsonl` have been built successfully with `8000` train records and `2000` validation records
- a GOT-OCR2.0 LoRA smoke training run has succeeded with `sdpa`
- single-GPU training is the currently verified stable baseline route on this machine
- the next model step is to train a fresh LoRA checkpoint on the v1 synthetic Swift manifests

The current verified status as of 2026-04-13 additionally includes:
- real-test PDF extraction ran successfully in the `rapid` environment
- the `2006 流量` PDF rendered to `18` pages and yielded `35` station-level table crops
- the `2014 水位` PDF rendered to `24` pages and yielded `47` station-level table crops
- the current extraction run produced `0` logged layout failures
- extracted real-test artifacts now include `station_tables_plain/`, `station_tables_buffered/`, and `title_rois/`
- title OCR uses a dedicated ROI built from the upper buffer plus a small downward compensation
- successful titles are cleaned to the prefix before `逐日平均流量表` or `逐日平均水位表`
- final extracted filenames now use the form `稳定ID_标题_年份`

The current verified status as of 2026-04-17 additionally includes:
- `station_tables_daily/` is the official OCR-cropped real-test image set for `2006 流量`
- the default real-data alignment path now uses `datasets/build_real_flow_alignment.py`
- the current aligned `2006 流量` audit reports `35` CSV files, `35` images, and `35` confirmed matches
- real-data entrypoints remain under `datasets/`, while shared implementation now lives under `src/yearbook_ocr/`

The current verified status as of 2026-04-19 additionally includes:
- `2014 水位` daily-only OCR crops have been regenerated successfully with `47` input tables, `47` cropped outputs, and `0` failures
- the `2014 水位` crop path now uses the same lower-left statistics ROI as `2006 流量`
- anchor selection now prefers full `平均`, then falls back to constrained `平/均` weak anchors inside the statistics label column
- `2014 水位` metadata currently reports `28` `average_anchor` crops and `19` `weak_average_anchor` crops
- `year_stats_anchor` is no longer used in the formal real-data crop path

## Current Scope

The current direction is intentionally narrow:
- focus on `流量` only
- keep `水位` out of the current training and final evaluation scope
- use one fixed flow-table layout
- use synthetic data for train and validation
- use real calibrated flow tables for final test only
- keep calibrated CSV as label authority, while the current GOT baseline is trained against official GOT formatted table text

## Current Gaps

The main remaining execution gaps are:
- train the v1 LoRA model from the current synthetic Swift manifests
- rerun base and fine-tuned inference on the real extracted flow tables
- run a formal base-vs-v1-fine-tuned comparison on the real extracted flow tables
- continue improving the raw GOT/LaTeX prediction benchmark on the aligned real flow tables

The main remaining real-data gap outside the current flow benchmark is:
- `2014 水位` still lacks calibrated CSV labels in the repository, so no alignment manifest or raw-output benchmark has been built for it yet

## Current Recommended Training Route

The current stable route is:
- use the `got` conda environment
- use `ms-swift + LoRA`
- use `sdpa`, not `flash-attn`
- use project-local caches under `outputs/cache/`
- launch through `scripts/models/got_ocr2/run_swift_sft.sh`
- use single-GPU training as the official baseline route on this machine
- only treat multi-GPU as experimental for now
- use `scripts/models/got_ocr2/run_inference.py` as the only formal GOT inference entrypoint

Important environment findings from the verified smoke run:
- `bitsandbytes==0.41.0` blocked `swift` startup through `peft`
- older `peft==0.4.0` was not compatible with the current `ms-swift` setup
- `flash-attn==2.8.3` failed to load because its binary required `GLIBC_2.32` while the system provides `2.31`
- `device_map=auto` did not actually shard GOT-OCR2.0 across all GPUs in this environment
- `LoRA + gradient checkpointing + DDP(device_map)` triggered a backward compatibility error
- explicit ZeRO2 configuration was parsed successfully, but 8-GPU smoke runs still failed to enter stable training on this machine

## Data Notes

- Calibrated CSV files are the source-of-truth labels.
- Existing CSV files may use local Chinese encodings rather than UTF-8.
- Raw calibrated CSV files and filenames should be preserved.
- Manifest targets remove empty artifact rows and trailing empty artifact columns but preserve empty cells inside non-empty rows and natural calendar blanks.
- Existing filenames encode station, year, and river context and should be preserved.

## Code Organization

The current code roles are:
- `datasets/`: source data, derived assets, and thin real-data entrypoints
- `src/yearbook_ocr/`: shared implementation for data, OCR, GOT inference, and evaluation
- `scripts/data/`: synthetic generation entrypoints
- `scripts/models/`: thin model-specific adapters and wrappers
- `scripts/eval/`: thin evaluation wrappers
- `configs/`: experiment configuration
- `tests/`: utility and metric tests

## Documentation Strategy

Document responsibilities are split as follows:
- `README.md`: public-facing project homepage
- `README.zh-CN.md`: Chinese-facing homepage
- `AGENTS.md`: short contributor and agent entry point
- `docs/`: detailed project notes, rules, and workflow documents
