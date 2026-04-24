# Hydro Yearbook OCR VLM

[中文](README.zh-CN.md)

A repository for fine-tuning and evaluating vision-language models on hydrological yearbook OCR.

## Overview

This project targets OCR on hydrological yearbook flow tables. The current v0 training and evaluation scope is intentionally narrow: one fixed flow-table layout, one model family, one task contract.

The v0 task is:
- single table image -> raw CSV text

The current baseline direction is:
- model: `GOT-OCR2.0`
- training: synthetic data first
- final test: real calibrated flow tables only
- evaluation: strict raw-output scoring with no post-processing

## Current Scope

Current v0 decisions:
- focus on `流量` only
- keep `水位` out of the v0 training and final-evaluation scope
- use one fixed flow-table layout
- render synthetic tables programmatically instead of reusing real screenshots
- preserve natural calendar blanks and a small amount of missing values
- prioritize geometric and lighting perturbations
- use `ms-swift + LoRA` as the preferred training path

## Current Status

Repository status as of 2026-04-17:
- source PDFs and calibrated CSV labels are stored under `datasets/`
- calibrated CSV files are treated as source-of-truth labels
- the main documentation set under `docs/` is in place
- v0 synthetic data generation is implemented and smoke-tested
- real flow test manifest generation is implemented and smoke-tested
- real PDF page rendering and layout-based station-table extraction are implemented
- title-aware real-test extraction with `plain`, `buffered`, and `title_rois` outputs is implemented
- OCR-based daily-only crop generation for real `2006 流量` tables is implemented
- scored alignment from extracted daily crops to calibrated `2006 流量` CSV labels is implemented
- OCR-based daily-only crop generation for real `2014 水位` tables is also implemented
- the real-data crop path now uses a dedicated lower-left statistics ROI and accepts `平均` first, then constrained `平/均` weak anchors when full `平均` is missed
- `ms-swift` manifest conversion for `GOT-OCR2.0` is implemented and smoke-tested
- strict CSV evaluation is implemented and covered by tests
- a Linux environment setup script is included
- a `swift sft` wrapper for `GOT-OCR2.0` is included and validated on a smoke run
- small sample outputs and manifest previews already exist under `data/`
- `conda run -n got pytest -q` passes with `6 passed`
- a GOT-OCR2.0 LoRA smoke run already succeeded with `sdpa`

Still missing:
- the first full multi-GPU `GOT-OCR2.0` LoRA training run
- final inference and strict evaluation on real extracted table images

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
- `datasets/流量/` contains the current v0 target data
- `datasets/水位/` is present and has validated real-table extraction and daily crop outputs, but it is still out of scope for the v0 model benchmark
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
├── datasets/                 # Source PDFs and calibrated CSV labels
├── docs/                     # Project notes and detailed specifications
├── references/               # External reference code and papers
├── datasets/*.py             # Real-data extraction, crop, and alignment utilities
├── scripts/common/           # Shared flow-table utilities
├── scripts/data/             # Data prep and synthetic generation
├── scripts/eval/             # Evaluation
├── scripts/models/           # Model-specific training adapters
├── tests/                    # Automated tests
├── AGENTS.md                 # Minimal contributor/agent guide
├── README.md                 # English homepage
└── README.zh-CN.md           # Chinese homepage
```

## Quick Start

Use the Linux `got` conda environment for this project.

1. Read [docs/project-overview.md](docs/project-overview.md)
2. Review the v0 data contract in [docs/data-spec.md](docs/data-spec.md)
3. Check environment requirements in [docs/environment-setup.md](docs/environment-setup.md)
4. Inspect the calibrated CSV labels under `datasets/流量/2006/`
5. Review real-test extraction in [docs/real-test-extraction.md](docs/real-test-extraction.md)

Representative commands:

```bash
conda run -n rapid python ./datasets/crop_flow_table_daily_region.py
conda run -n rapid python ./datasets/run_ocr_on_image.py 'datasets/derived/debug_rois/2014_水位/page_0004_table1_汉江汉川站_2014.jpg' --ocr-cuda off
python3 ./datasets/build_real_flow_alignment.py
python3 ./scripts/data/generate_synthetic_flow_v0.py --num-samples 10000
python3 ./scripts/models/got_ocr2/build_swift_manifest.py --input data/manifests/flow_v0/train.jsonl --output data/manifests/flow_v0/train_swift.jsonl
python3 ./scripts/models/got_ocr2/build_swift_manifest.py --input data/manifests/flow_v0/val.jsonl --output data/manifests/flow_v0/val_swift.jsonl
python3 ./scripts/models/got_ocr2/run_inference.py --manifest data/manifests/flow_real_test_aligned.jsonl --limit 5
python3 ./scripts/eval/evaluate_strict_csv.py --predictions outputs/predictions/got_ocr2_v0.jsonl --output outputs/reports/got_ocr2_v0_eval.json
```

These scripts are implemented, but they require a suitable Python 3.10+ environment and the training stack is not yet validated end to end on this machine.
These scripts are implemented. The current machine has already passed data-prep, tests, and a 1-step GOT-OCR2.0 LoRA smoke run in the `got` environment.

## Roadmap

Near-term execution order:
- launch the first full `GOT-OCR2.0` LoRA fine-tuning run
- evaluate on the real flow test set with strict raw-output scoring
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
