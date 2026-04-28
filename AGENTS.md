# Repository Guidelines

This repository is for fine-tuning and evaluating a VLM on hydrological yearbook OCR tasks.

## Where To Read First
- [README.md](/home/yubin/py_prj/yearbook_VLM/README.md): public project overview, status, and roadmap
- [docs/project-overview.md](/home/yubin/py_prj/yearbook_VLM/docs/project-overview.md): internal project context and current data status
- `datasets/`: source PDFs and calibrated CSV labels

## Docs Index
- [docs/data-spec.md](/home/yubin/py_prj/yearbook_VLM/docs/data-spec.md): manifest schema, target cleanup rules, label authority, and layout JSON contract
- [docs/environment-setup.md](/home/yubin/py_prj/yearbook_VLM/docs/environment-setup.md): `got` and `rapid` environment setup, dependency notes, and verification commands
- [docs/experiment-plan.md](/home/yubin/py_prj/yearbook_VLM/docs/experiment-plan.md): benchmark goal, fixed decisions, metrics, and current execution order
- [docs/got-ocr2-finetune.md](/home/yubin/py_prj/yearbook_VLM/docs/got-ocr2-finetune.md): GOT-OCR2.0 Swift/LoRA training, inference, cache, and evaluation workflow
- [docs/real-test-extraction.md](/home/yubin/py_prj/yearbook_VLM/docs/real-test-extraction.md): real PDF table extraction, daily crop generation, and alignment workflow
- [docs/running_command.md](/home/yubin/py_prj/yearbook_VLM/docs/running_command.md): copy-ready commands for data generation, training, inference, evaluation, and git
- [docs/synthetic-data.md](/home/yubin/py_prj/yearbook_VLM/docs/synthetic-data.md): synthetic rendering rules, v1/v2 generation settings, and data-regime notes

## Working Rules
- Keep `AGENTS.md` minimal. Do not duplicate material that already belongs in `README.md` or `docs/`.
- Add detailed specifications, data rules, and workflow notes under `docs/`.
- Treat calibrated CSV files as source-of-truth labels and preserve existing dataset filenames.
- Use the `got` conda environment for this project.
- Keep data-generation code and model-specific code in separate directories.
- Keep dataset preprocessing, extraction, alignment, and crop-generation code under `datasets/` rather than `scripts/`.
- Prefer one parameterized inference/eval entrypoint per model workflow rather than adding overlapping one-off scripts for single-image, small-batch, and full-set runs.
- When a temporary probe script proves useful, consolidate it back into the main inference/eval entrypoint instead of letting multiple near-duplicate scripts accumulate.
- Do not change the main algorithm for real-data preprocessing without explicit user confirmation; fallback or experimental paths must stay opt-in.
- Do not ask the agent to run `git push` for this repository; the terminal may not have GitHub credentials configured. Remind the user to run `git push origin <branch>` manually when needed.
- Commit messages must follow this format: one summary line, then a blank line, then flat bullet summaries for the key changes.
