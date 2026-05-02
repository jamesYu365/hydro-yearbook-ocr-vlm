from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from yearbook_ocr.common.jsonl import load_jsonl
from yearbook_ocr.eval.model_comparison import score_structure_cell


FOLDERS = (
    "structure_correct__cell_correct",
    "structure_correct__cell_error",
    "structure_error__cell_correct",
    "structure_error__cell_error",
)


@dataclass
class ClassifiedOutput:
    sample_id: str
    image_path: str
    raw_path: str
    copied_path: str
    folder: str
    structure_correct: bool
    cell_correct: bool
    cell_accuracy: float
    cell_correct_count: int
    total_cell_count: int
    error_type: str


def raw_output_path(raw_dir: Path, record: dict[str, Any]) -> Path:
    return raw_dir / (Path(record["image_path"]).stem + ".txt")


def classify_record(record: dict[str, Any], raw_dir: Path, output_dir: Path) -> ClassifiedOutput:
    metrics = score_structure_cell(record)
    cell_correct = metrics.structure_correct and metrics.cell_correct_count == metrics.total_cell_count
    folder = (
        ("structure_correct" if metrics.structure_correct else "structure_error")
        + "__"
        + ("cell_correct" if cell_correct else "cell_error")
    )
    source = raw_output_path(raw_dir, record)
    if not source.exists():
        raise FileNotFoundError(f"Missing per-image raw output for {record['sample_id']}: {source}")

    destination = output_dir / folder / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)

    return ClassifiedOutput(
        sample_id=record["sample_id"],
        image_path=record.get("image_path", ""),
        raw_path=source.as_posix(),
        copied_path=destination.as_posix(),
        folder=folder,
        structure_correct=metrics.structure_correct,
        cell_correct=cell_correct,
        cell_accuracy=metrics.cell_accuracy,
        cell_correct_count=metrics.cell_correct_count,
        total_cell_count=metrics.total_cell_count,
        error_type=metrics.error_type,
    )


def write_summary(rows: list[ClassifiedOutput], output_dir: Path) -> dict[str, Any]:
    counts = {folder: 0 for folder in FOLDERS}
    for row in rows:
        counts[row.folder] += 1

    summary = {
        "sample_count": len(rows),
        "folder_counts": counts,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with (output_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
    return summary


def classify_outputs(predictions: Path, raw_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for folder in FOLDERS:
        (output_dir / folder).mkdir(parents=True, exist_ok=True)

    rows = [
        classify_record(record, raw_dir=raw_dir, output_dir=output_dir)
        for record in load_jsonl(predictions)
    ]
    return write_summary(rows, output_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify per-image raw GOT outputs by structure/cell correctness.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = classify_outputs(args.predictions, args.raw_dir, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
