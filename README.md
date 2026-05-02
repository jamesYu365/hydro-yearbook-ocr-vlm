# Hydro Yearbook OCR VLM

[中文](README.zh-CN.md)

A repository for fine-tuning and evaluating vision-language models on hydrological yearbook OCR.

## Overview

This project targets OCR on hydrological yearbook flow tables. The current v1 training and evaluation scope is intentionally narrow: one fixed flow-table layout, one model family, one task contract.

The current repository tracks two related task views:
- label authority: calibrated CSV remains the source-of-truth representation
- current GOT baseline target: single table image -> official GOT formatted table text

The current baseline direction is:
- model: `GOT-OCR2.0`
- training: synthetic data first
- final test: real calibrated flow tables only
- training target: official GOT `OCR with format:` output
- evaluation: score raw/pretty GOT table output with the shared table parser against CSV source-of-truth targets

## Current Scope

Current v1 decisions:
- focus on `流量` only
- keep `水位` out of the current training and final-evaluation scope
- use one fixed flow-table layout
- render synthetic tables programmatically instead of reusing real screenshots
- preserve natural calendar blanks and a small amount of missing values
- omit empty artifact rows and trailing empty artifact columns from manifest targets while keeping source CSV files unchanged
- prioritize geometric and lighting perturbations
- use `ms-swift + LoRA` as the preferred training path

## Current Status

Repository status as of 2026-04-24:
- source PDFs and calibrated CSV labels are stored under `datasets/`
- calibrated CSV files are treated as source-of-truth labels
- the main documentation set under `docs/` is in place
- v1 synthetic data generation is implemented with real-crop-like proportions, diagonal day/month header rendering, and strict shuffle split
- real flow test manifest generation is implemented and smoke-tested
- real PDF page rendering and layout-based station-table extraction are implemented
- title-aware real-test extraction with `plain`, `buffered`, and `title_rois` outputs is implemented
- OCR-based daily-only crop generation for real `2006 流量` tables is implemented
- scored alignment from extracted daily crops to calibrated `2006 流量` CSV labels is implemented
- OCR-based daily-only crop generation for real `2014 水位` tables is also implemented
- the real-data crop path now uses a dedicated lower-left statistics ROI and accepts `平均` first, then constrained `平/均` weak anchors when full `平均` is missed
- `ms-swift` manifest conversion for `GOT-OCR2.0` is implemented and smoke-tested
- a shared `src/yearbook_ocr/` package now holds stable data, OCR, inference, and evaluation logic
- unified GOT inference now runs through `scripts/models/got_ocr2/run_inference.py`
- table-format evaluation for raw GOT/LaTeX output is implemented and covered by tests
- a Linux environment setup script is included
- a `swift sft` wrapper for `GOT-OCR2.0` is included and validated on a smoke run
- small sample outputs and manifest previews already exist under `data/`
- focused data, manifest, inference, and evaluation test suites pass in `got`
- a GOT-OCR2.0 LoRA smoke run already succeeded with `sdpa`
- current v1 Swift manifests contain `8000` train records and `2000` validation records
- the aligned real flow test manifest contains `35` records

Still missing:
- a fresh `GOT-OCR2.0` LoRA run on the v1 synthetic manifests
- fresh base and fine-tuned inference runs on the real extracted flow test set
- a formal benchmark report comparing base GOT vs the v1 fine-tuned GOT model

## Verified Stable Path

The current recommended path is:
- conda env: `got`
- training method: `ms-swift + LoRA`
- attention backend: `sdpa`
- cache root: `outputs/cache/`
- wrapper entrypoint: `scripts/models/got_ocr2/run_swift_sft.sh`

Known environment findings from the validated smoke run:
- do not rely on `flash-attn` on this machine
- do not use `bitsandbytes==0.41.0` for this path
- older `peft==0.4.0` is not compatible with the current `ms-swift` setup

## Data Snapshot

Data notes:
- `datasets/流量/` contains the current target data
- `datasets/水位/` is present and has validated real-table extraction and daily crop outputs, but it is still out of scope for the current model benchmark
- `datasets/derived/` stores generated real-test extraction outputs
- extracted real-test images now use the form `稳定ID_标题_年份`
- `station_tables_daily/` is the current official OCR-cropped real-test image set for `2006 流量`
- `station_tables_daily/` is also available for `2014 水位`, produced by the same ROI-based OCR crop path
- some CSV files use local Chinese encodings rather than UTF-8
- original filenames encode station, year, and river context and should be preserved

## Repository Structure

```text
.
├── configs/                  # Experiment configuration
├── data/                     # Generated sample outputs and manifests
├── datasets/                 # Source PDFs, calibrated CSV labels, and thin real-data entrypoints
├── docs/                     # Project notes and detailed specifications
├── references/               # External reference code and papers
├── src/yearbook_ocr/         # Shared package for data, OCR, GOT inference, and evaluation
├── scripts/data/             # Synthetic generation entrypoints
├── scripts/eval/             # Thin evaluation wrappers
├── scripts/models/           # Thin model-specific wrappers
├── tests/                    # Automated tests
├── AGENTS.md                 # Minimal contributor/agent guide
├── README.md                 # English homepage
└── README.zh-CN.md           # Chinese homepage
```

## Quick Start

Use the Linux `got` conda environment for this project.

1. Read [docs/project-overview.md](docs/project-overview.md)
2. Review the current data contract in [docs/data-spec.md](docs/data-spec.md)
3. Check environment requirements in [docs/environment-setup.md](docs/environment-setup.md)
4. Inspect the calibrated CSV labels under `datasets/流量/2006/`
5. Review real-test extraction in [docs/real-test-extraction.md](docs/real-test-extraction.md)

Representative commands:

```bash
conda run -n rapid python ./datasets/crop_flow_table_daily_region.py
conda run -n rapid python ./datasets/run_ocr_on_image.py 'datasets/derived/debug_rois/2014_水位/page_0004_table1_汉江汉川站_2014.jpg' --ocr-cuda off
python3 ./datasets/build_real_flow_alignment.py
python3 ./scripts/data/generate_synthetic_flow_v0.py --num-samples 10000 --val-ratio 0.2 --seed 20260408 --num-workers 16
python3 ./scripts/models/got_ocr2/build_swift_manifest.py --input data/manifests/flow_v0/train.jsonl --output data/manifests/flow_v0/train_swift.jsonl
python3 ./scripts/models/got_ocr2/build_swift_manifest.py --input data/manifests/flow_v0/val.jsonl --output data/manifests/flow_v0/val_swift.jsonl
python3 ./scripts/models/got_ocr2/run_inference.py --manifest data/manifests/flow_real_test_aligned.jsonl --limit 5 --backend official_chat --query-mode official_format
python3 ./scripts/eval/evaluate_predictions.py --predictions outputs/got_ocr2_base/eval/checkpoint-0/flow_real_first5_official_chat.jsonl --output outputs/reports/got_ocr2_base_first5_eval.json
```

These scripts are implemented. The current machine has already passed data-prep, focused tests, and a 1-step GOT-OCR2.0 LoRA smoke run in the `got` environment.

## Roadmap

Near-term execution order:
- train `GOT-OCR2.0` LoRA from the v1 synthetic Swift manifests
- rerun base and fine-tuned inference on the aligned real flow test set
- run table-format evaluation and model comparison on raw/pretty GOT outputs
- inspect representative failure cases and decide the next iteration

## Documentation

- [Project Overview](docs/project-overview.md)
- [Data Spec](docs/data-spec.md)
- [Synthetic Data](docs/synthetic-data.md)
- [Experiment Plan](docs/experiment-plan.md)
- [Environment Setup](docs/environment-setup.md)
- [Real Test Extraction](docs/real-test-extraction.md)
- [GOT-OCR2.0 Fine-tuning](docs/got-ocr2-finetune.md)
- [Contributor Guide](AGENTS.md)

## Contributing

Keep detailed project notes under `docs/` and keep `AGENTS.md` minimal.

## License

License information has not been added yet.
