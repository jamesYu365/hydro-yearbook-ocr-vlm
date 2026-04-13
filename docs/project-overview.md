# Project Overview

## Summary

Hydro Yearbook OCR VLM is a focused project for fine-tuning a vision-language model on hydrological yearbook OCR, starting from fixed-layout flow tables.

The immediate v0 objective is:
- fine-tune `GOT-OCR2.0`
- train mainly on high-fidelity synthetic flow tables
- evaluate on real calibrated flow tables
- score raw model output as CSV with no post-processing

## Current Repository State

The repository is no longer documentation-only. The current implemented pieces are:
- source PDFs and calibrated CSV labels under `datasets/`
- real-test PDF page rendering and layout-based station-table extraction under `datasets/`
- shared flow-table utilities under `scripts/common/`
- synthetic flow-table generation under `scripts/data/`
- real test manifest generation under `scripts/data/`
- `ms-swift` manifest conversion for `GOT-OCR2.0` under `scripts/models/got_ocr2/`
- strict CSV evaluation under `scripts/eval/`
- a Linux environment setup script under `scripts/`
- basic tests under `tests/`
- a validated `swift sft` wrapper under `scripts/models/got_ocr2/`

Existing generated artifacts under `data/` already include:
- small synthetic sample images
- layout JSON files
- synthetic manifests
- a real test manifest
- a `ms-swift` manifest preview

The current verified status as of 2026-04-08 is:
- `conda run -n got pytest -q` passes with `6 passed`
- synthetic data generation smoke test has passed
- real test manifest generation has passed with 35 records
- `train_swift.jsonl` and `val_swift.jsonl` have been built successfully
- a GOT-OCR2.0 LoRA smoke training run has succeeded with `sdpa`
- single-GPU training is the currently verified stable baseline route on this machine

The current verified status as of 2026-04-13 additionally includes:
- real-test PDF extraction ran successfully in the `rapid` environment
- the `2006 流量` PDF rendered to `18` pages and yielded `35` station-level table crops
- the `2014 水位` PDF rendered to `24` pages and yielded `47` station-level table crops
- the current extraction run produced `0` logged layout failures

## v0 Scope

The current v0 direction is intentionally narrow:
- focus on `流量` only
- do not handle `水位` in v0
- use one fixed flow-table layout
- use synthetic data for train and validation
- use real calibrated flow tables for final test only
- keep the task fixed as single table image to raw CSV text

## Current Gaps

The main remaining execution gaps are:
- launch the first full baseline training run and collect the first durable checkpoints
- align extracted station-table images with the calibrated labels
- run final inference and strict evaluation on real extracted table images

## Current Recommended Training Route

The current stable route is:
- use the `got` conda environment
- use `ms-swift + LoRA`
- use `sdpa`, not `flash-attn`
- use project-local caches under `outputs/cache/`
- launch through `scripts/models/got_ocr2/run_swift_sft.sh`
- use single-GPU training as the official baseline route on this machine
- only treat multi-GPU as experimental for now

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
- Empty separator rows in the CSV should be preserved because they reflect the original table structure.
- Existing filenames encode station, year, and river context and should be preserved.

## Code Organization

The current code roles are:
- `datasets/`: source data plus real-test extraction utilities and derived extraction outputs
- `scripts/data/`: synthetic generation and real test manifest building
- `scripts/models/`: model-specific adapters and training wrappers
- `scripts/eval/`: strict evaluation
- `scripts/common/`: shared parsing, encoding, and prompt utilities
- `configs/`: experiment configuration
- `tests/`: utility and metric tests

## Documentation Strategy

Document responsibilities are split as follows:
- `README.md`: public-facing project homepage
- `README.zh-CN.md`: Chinese-facing homepage
- `AGENTS.md`: short contributor and agent entry point
- `docs/`: detailed project notes, rules, and workflow documents
