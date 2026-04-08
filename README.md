# Hydro Yearbook OCR VLM

[中文](README.zh-CN.md)

A repository for fine-tuning and evaluating vision-language models on hydrological yearbook OCR.

## Overview

This project targets OCR on hydrological yearbook flow tables. The current v0 scope is intentionally narrow: one fixed table layout, one model family, one task contract.

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
- do not handle `水位` yet
- use one fixed flow-table layout
- render synthetic tables programmatically instead of reusing real screenshots
- preserve natural calendar blanks and a small amount of missing values
- prioritize geometric and lighting perturbations
- use `ms-swift + LoRA` as the preferred training path

## Current Status

Repository status as of the current Linux migration:
- source PDFs and calibrated CSV labels are stored under `datasets/`
- calibrated CSV files are treated as source-of-truth labels
- the main documentation set under `docs/` is in place
- v0 synthetic data generation is scaffolded
- real flow test manifest generation is scaffolded
- `ms-swift` manifest conversion for `GOT-OCR2.0` is scaffolded
- strict CSV evaluation is implemented
- a Linux environment setup script is included
- a `swift sft` wrapper for `GOT-OCR2.0` is included
- small sample outputs and manifest previews already exist under `data/`

Still missing:
- full Linux training environment validation in the `got` conda environment
- real PDF to single-table image extraction for the final test set
- the first end-to-end `GOT-OCR2.0` LoRA fine-tuning run

## Data Snapshot

Data notes:
- `datasets/流量/` contains the current v0 target data
- `datasets/水位/` is present but out of scope for v0
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

Representative commands:

```bash
python3 ./scripts/data/build_real_test_manifest.py
python3 ./scripts/data/generate_synthetic_flow_v0.py --num-samples 10000
python3 ./scripts/models/got_ocr2/build_swift_manifest.py --input data/manifests/flow_v0/train.jsonl --output data/manifests/flow_v0/train_swift.jsonl
python3 ./scripts/models/got_ocr2/build_swift_manifest.py --input data/manifests/flow_v0/val.jsonl --output data/manifests/flow_v0/val_swift.jsonl
python3 ./scripts/eval/evaluate_strict_csv.py --predictions outputs/predictions/got_ocr2_v0.jsonl --output outputs/reports/got_ocr2_v0_eval.json
```

These scripts are implemented, but they require a suitable Python 3.10+ environment and the training stack is not yet validated end to end on this machine.

## Roadmap

Near-term execution order:
- validate the Linux `got` environment and run a smoke test
- extract real PDF pages into single-table station images
- launch the first `GOT-OCR2.0` LoRA fine-tuning run
- evaluate on the real flow test set with strict raw-output scoring
- inspect representative failure cases and decide the next iteration

## Documentation

- [Project Overview](docs/project-overview.md)
- [Data Spec](docs/data-spec.md)
- [Synthetic Data](docs/synthetic-data.md)
- [Experiment Plan](docs/experiment-plan.md)
- [Environment Setup](docs/environment-setup.md)
- [GOT-OCR2.0 Fine-tuning](docs/got-ocr2-finetune.md)
- [Contributor Guide](AGENTS.md)

## Contributing

Keep detailed project notes under `docs/` and keep `AGENTS.md` minimal.

## License

License information has not been added yet.
