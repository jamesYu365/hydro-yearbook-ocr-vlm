from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from yearbook_ocr.common.jsonl import load_jsonl
from yearbook_ocr.eval.csv_eval import classify_error


@dataclass
class StructureCellMetrics:
    sample_id: str
    image_path: str
    structure_correct: bool
    cell_accuracy: float
    cell_correct_count: int
    total_cell_count: int
    error_type: str


def parse_csv_strict(text: str) -> list[list[str]] | None:
    try:
        return list(csv.reader(text.splitlines(), strict=True))
    except csv.Error:
        return None


def parse_latex_tabular(text: str) -> list[list[str]] | None:
    if "\\begin{tabular}" not in text:
        return None
    rows: list[list[str]] = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("\\begin{tabular}") or line.startswith("\\end{tabular}"):
            continue
        line = line.replace("\\hline", "").strip()
        if not line:
            continue
        line = re.sub(r"\\\\\s*$", "", line).strip()
        if not line:
            continue
        rows.append([cell.strip() for cell in line.split("&")])
    return rows


def parse_table_rows(text: str) -> list[list[str]] | None:
    latex_rows = parse_latex_tabular(text)
    if latex_rows is not None:
        return latex_rows
    return parse_csv_strict(text)


def same_shape(left: list[list[str]], right: list[list[str]]) -> bool:
    return len(left) == len(right) and all(len(left_row) == len(right_row) for left_row, right_row in zip(left, right))


def count_target_cells(rows: list[list[str]]) -> int:
    return sum(len(row) for row in rows)


def score_structure_cell(record: dict[str, Any]) -> StructureCellMetrics:
    sample_id = record["sample_id"]
    prediction = record["prediction"]
    target = record["target_csv"]
    pred_rows = parse_table_rows(prediction)
    target_rows = parse_table_rows(target)
    if target_rows is None:
        raise ValueError(f"Target CSV is not parseable for {sample_id}")

    total_cells = count_target_cells(target_rows)
    structure_correct = pred_rows is not None and same_shape(pred_rows, target_rows)
    correct_cells = 0
    if structure_correct and pred_rows is not None:
        correct_cells = sum(
            1
            for pred_row, target_row in zip(pred_rows, target_rows)
            for pred_cell, target_cell in zip(pred_row, target_row)
            if pred_cell == target_cell
        )
    return StructureCellMetrics(
        sample_id=sample_id,
        image_path=record.get("image_path", ""),
        structure_correct=structure_correct,
        cell_accuracy=correct_cells / max(total_cells, 1),
        cell_correct_count=correct_cells,
        total_cell_count=total_cells,
        error_type=classify_error(prediction, target, pred_rows, target_rows),
    )


def index_records(records: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for record in records:
        sample_id = record["sample_id"]
        if sample_id in indexed:
            duplicates.append(sample_id)
        indexed[sample_id] = record
    if duplicates:
        duplicate_list = ", ".join(sorted(set(duplicates)))
        raise ValueError(f"{label} predictions contain duplicate sample_id values: {duplicate_list}")
    return indexed


def compare_sample_sets(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    before_ids = set(before)
    after_ids = set(after)
    missing_after = sorted(before_ids - after_ids)
    missing_before = sorted(after_ids - before_ids)
    if missing_after or missing_before:
        details = []
        if missing_after:
            details.append(f"missing from after: {', '.join(missing_after[:10])}")
        if missing_before:
            details.append(f"missing from before: {', '.join(missing_before[:10])}")
        raise ValueError("Prediction files do not contain the same sample_ids (" + "; ".join(details) + ")")
    return sorted(before_ids)


def summarize(label: str, metrics: list[StructureCellMetrics]) -> dict[str, Any]:
    sample_count = len(metrics)
    structure_correct_count = sum(1 for item in metrics if item.structure_correct)
    cell_correct_count = sum(item.cell_correct_count for item in metrics)
    total_cell_count = sum(item.total_cell_count for item in metrics)
    return {
        "label": label,
        "sample_count": sample_count,
        "structure_accuracy": structure_correct_count / max(sample_count, 1),
        "cell_accuracy": cell_correct_count / max(total_cell_count, 1),
        "structure_correct_count": structure_correct_count,
        "cell_correct_count": cell_correct_count,
        "total_cell_count": total_cell_count,
        "by_error_type": dict(Counter(item.error_type for item in metrics)),
    }


def build_sample_rows(
    sample_ids: list[str],
    before_metrics: dict[str, StructureCellMetrics],
    after_metrics: dict[str, StructureCellMetrics],
) -> list[dict[str, Any]]:
    rows = []
    for sample_id in sample_ids:
        before = before_metrics[sample_id]
        after = after_metrics[sample_id]
        rows.append(
            {
                "sample_id": sample_id,
                "image_path": before.image_path or after.image_path,
                "before_structure_correct": before.structure_correct,
                "after_structure_correct": after.structure_correct,
                "structure_delta": int(after.structure_correct) - int(before.structure_correct),
                "before_cell_accuracy": before.cell_accuracy,
                "after_cell_accuracy": after.cell_accuracy,
                "cell_accuracy_delta": after.cell_accuracy - before.cell_accuracy,
                "before_error_type": before.error_type,
                "after_error_type": after.error_type,
                "error_type_transition": f"{before.error_type} -> {after.error_type}",
                "before_cell_correct_count": before.cell_correct_count,
                "after_cell_correct_count": after.cell_correct_count,
                "total_cell_count": before.total_cell_count,
            }
        )
    return rows


def build_structure_transitions(sample_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    for row in sample_rows:
        before = "correct" if row["before_structure_correct"] else "incorrect"
        after = "correct" if row["after_structure_correct"] else "incorrect"
        counts[f"{before} -> {after}"] += 1
    return [{"transition": key, "count": counts[key]} for key in sorted(counts)]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_plots(output_dir: Path, before_summary: dict[str, Any], after_summary: dict[str, Any], sample_rows: list[dict[str, Any]]) -> None:
    mpl_config_dir = Path(os.environ.get("MPLCONFIGDIR", "outputs/cache/matplotlib"))
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", mpl_config_dir.as_posix())

    import matplotlib.pyplot as plt

    labels = [before_summary["label"], after_summary["label"]]
    colors = ["#8A9199", "#7F8F84"]

    for metric_name, filename, title in [
        ("structure_accuracy", "structure_accuracy.png", "Structure Accuracy"),
        ("cell_accuracy", "cell_accuracy.png", "Cell Accuracy"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4))
        values = [before_summary[metric_name], after_summary[metric_name]]
        ax.bar(labels, values, color=colors)
        ax.set_ylim(0, 1)
        ax.set_ylabel(metric_name)
        ax.set_title(title)
        for index, value in enumerate(values):
            ax.text(index, value + 0.02, f"{value:.3f}", ha="center", va="bottom")
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist([row["cell_accuracy_delta"] for row in sample_rows], bins=11, color="#B7A99A", edgecolor="#5F5A52")
    ax.axvline(0, color="#5F5A52", linewidth=1)
    ax.set_xlabel("after - before cell accuracy")
    ax.set_ylabel("sample count")
    ax.set_title("Per-Sample Cell Accuracy Delta")
    fig.tight_layout()
    fig.savefig(output_dir / "cell_accuracy_delta_hist.png", dpi=160)
    plt.close(fig)


def compare_predictions(
    before_predictions: Path,
    after_predictions: Path,
    before_label: str,
    after_label: str,
    output_dir: Path,
    *,
    write_images: bool = True,
) -> dict[str, Any]:
    before_records = index_records(load_jsonl(before_predictions), before_label)
    after_records = index_records(load_jsonl(after_predictions), after_label)
    sample_ids = compare_sample_sets(before_records, after_records)

    before_metrics = {sample_id: score_structure_cell(before_records[sample_id]) for sample_id in sample_ids}
    after_metrics = {sample_id: score_structure_cell(after_records[sample_id]) for sample_id in sample_ids}
    sample_rows = build_sample_rows(sample_ids, before_metrics, after_metrics)
    before_summary = summarize(before_label, [before_metrics[sample_id] for sample_id in sample_ids])
    after_summary = summarize(after_label, [after_metrics[sample_id] for sample_id in sample_ids])
    transitions = build_structure_transitions(sample_rows)
    payload = {
        "before_predictions": before_predictions.as_posix(),
        "after_predictions": after_predictions.as_posix(),
        "before": before_summary,
        "after": after_summary,
        "delta": {
            "structure_accuracy": after_summary["structure_accuracy"] - before_summary["structure_accuracy"],
            "cell_accuracy": after_summary["cell_accuracy"] - before_summary["cell_accuracy"],
            "structure_correct_count": after_summary["structure_correct_count"] - before_summary["structure_correct_count"],
            "cell_correct_count": after_summary["cell_correct_count"] - before_summary["cell_correct_count"],
        },
        "structure_transitions": transitions,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(
        output_dir / "summary.csv",
        [
            before_summary,
            after_summary,
            {
                "label": "delta",
                "sample_count": "",
                "structure_accuracy": payload["delta"]["structure_accuracy"],
                "cell_accuracy": payload["delta"]["cell_accuracy"],
                "structure_correct_count": payload["delta"]["structure_correct_count"],
                "cell_correct_count": payload["delta"]["cell_correct_count"],
                "total_cell_count": "",
            },
        ],
        [
            "label",
            "sample_count",
            "structure_accuracy",
            "cell_accuracy",
            "structure_correct_count",
            "cell_correct_count",
            "total_cell_count",
        ],
    )
    write_csv(
        output_dir / "samples.csv",
        sample_rows,
        [
            "sample_id",
            "image_path",
            "before_structure_correct",
            "after_structure_correct",
            "structure_delta",
            "before_cell_accuracy",
            "after_cell_accuracy",
            "cell_accuracy_delta",
            "before_error_type",
            "after_error_type",
            "error_type_transition",
            "before_cell_correct_count",
            "after_cell_correct_count",
            "total_cell_count",
        ],
    )
    write_csv(output_dir / "structure_transitions.csv", transitions, ["transition", "count"])
    if write_images:
        write_plots(output_dir, before_summary, after_summary, sample_rows)
    return payload


def safe_label(label: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in label).strip("_")


def default_output_dir(before_label: str, after_label: str) -> Path:
    return Path("outputs/model_comparisons") / f"{safe_label(before_label)}_vs_{safe_label(after_label)}"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare structure accuracy and cell accuracy for two prediction JSONL files.")
    parser.add_argument("--before-predictions", type=Path, required=True)
    parser.add_argument("--after-predictions", type=Path, required=True)
    parser.add_argument("--before-label", type=str, default="before")
    parser.add_argument("--after-label", type=str, default="after")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG chart generation.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = args.output_dir or default_output_dir(args.before_label, args.after_label)
    payload = compare_predictions(
        before_predictions=args.before_predictions,
        after_predictions=args.after_predictions,
        before_label=args.before_label,
        after_label=args.after_label,
        output_dir=output_dir,
        write_images=not args.no_plots,
    )
    print(json.dumps({"output_dir": output_dir.as_posix(), "delta": payload["delta"]}, ensure_ascii=False, indent=2))
