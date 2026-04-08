# Repository Guidelines

This repository is for fine-tuning and evaluating a VLM on hydrological yearbook OCR tasks.

## Where To Read First
- [README.md](/home/yubin/py_prj/yearbook_VLM/README.md): public project overview, status, and roadmap
- [docs/project-overview.md](/home/yubin/py_prj/yearbook_VLM/docs/project-overview.md): internal project context and current data status
- `datasets/`: source PDFs and calibrated CSV labels

## Working Rules
- Keep `AGENTS.md` minimal. Do not duplicate material that already belongs in `README.md` or `docs/`.
- Add detailed specifications, data rules, and workflow notes under `docs/`.
- Treat calibrated CSV files as source-of-truth labels and preserve existing dataset filenames.
- Use the `got` conda environment for this project.
- Keep data-generation code and model-specific code in separate directories.
