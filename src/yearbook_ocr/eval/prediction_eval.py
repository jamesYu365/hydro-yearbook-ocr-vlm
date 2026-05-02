from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    prefix_len = 0
    max_prefix_len = min(len(left), len(right))
    while prefix_len < max_prefix_len and left[prefix_len] == right[prefix_len]:
        prefix_len += 1
    if prefix_len:
        left = left[prefix_len:]
        right = right[prefix_len:]
        if not left:
            return len(right)
        if not right:
            return len(left)
    suffix_len = 0
    max_suffix_len = min(len(left), len(right))
    while suffix_len < max_suffix_len and left[-suffix_len - 1] == right[-suffix_len - 1]:
        suffix_len += 1
    if suffix_len:
        left = left[:-suffix_len]
        right = right[:-suffix_len]
        if not left:
            return len(right)
        if not right:
            return len(left)
    if len(right) > len(left):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (left_char != right_char)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def character_accuracy(prediction: str, target: str) -> float:
    denominator = max(len(target), len(prediction), 1)
    return 1.0 - (levenshtein_distance(prediction, target) / denominator)


def parse_target_rows(text: str) -> list[list[str]]:
    try:
        return list(csv.reader(text.splitlines(), strict=True))
    except csv.Error as exc:
        raise ValueError("Target table text is not parseable") from exc


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


def parse_prediction_rows(text: str) -> list[list[str]] | None:
    return parse_latex_tabular(text)


def same_shape(left: list[list[str]], right: list[list[str]]) -> bool:
    return len(left) == len(right) and all(
        len(left_row) == len(right_row) for left_row, right_row in zip(left, right)
    )


def table_rows_to_text(rows: list[list[str]]) -> str:
    return "\n".join(",".join(row) for row in rows)


def has_prefix_rows(pred_rows: list[list[str]], target_rows: list[list[str]]) -> bool:
    if len(pred_rows) >= len(target_rows):
        return False
    return pred_rows == target_rows[: len(pred_rows)]


@dataclass
class SampleResult:
    sample_id: str
    weighted_score: float
    character_accuracy: float
    cell_accuracy: float
    error_type: str
    segment_scores: dict[str, float]


def classify_error(
    prediction: str,
    target: str,
    pred_rows: list[list[str]] | None,
    target_rows: list[list[str]],
) -> str:
    if pred_rows is None:
        return "structure_error"
    if has_prefix_rows(pred_rows, target_rows):
        return "truncation"
    if not same_shape(pred_rows, target_rows):
        return "structure_error"
    return "value_error"


def cell_accuracy(pred_rows: list[list[str]] | None, target_rows: list[list[str]]) -> float:
    if pred_rows is None or len(pred_rows) != len(target_rows):
        return 0.0
    total = 0
    correct = 0
    for pred_row, target_row in zip(pred_rows, target_rows):
        if len(pred_row) != len(target_row):
            return 0.0
        for pred_cell, target_cell in zip(pred_row, target_row):
            total += 1
            if pred_cell == target_cell:
                correct += 1
    return correct / max(total, 1)


def segment_row_ranges(total_rows: int) -> dict[str, tuple[int, int]]:
    first_end = max(1, round(total_rows * 0.25))
    last_start = min(total_rows, max(first_end, round(total_rows * 0.75)))
    return {
        "front_25": (0, first_end),
        "middle_50": (first_end, last_start),
        "back_25": (last_start, total_rows),
    }


def segment_scores(pred_rows: list[list[str]] | None, target_rows: list[list[str]]) -> dict[str, float]:
    ranges = segment_row_ranges(len(target_rows))
    scores: dict[str, float] = {}
    for name, (start, end) in ranges.items():
        target_slice = target_rows[start:end]
        if not target_slice:
            scores[name] = 0.0
            continue
        pred_slice = pred_rows[start:end] if pred_rows and len(pred_rows) >= end else None
        scores[name] = cell_accuracy(pred_slice, target_slice)
    return scores


def score_sample(
    sample_id: str,
    prediction: str,
    target: str,
) -> SampleResult:
    pred_rows = parse_prediction_rows(prediction)
    try:
        target_rows = parse_target_rows(target)
    except ValueError as exc:
        raise ValueError(f"Target table text is not parseable for {sample_id}") from exc
    if pred_rows is not None:
        char_acc = character_accuracy(table_rows_to_text(pred_rows), table_rows_to_text(target_rows))
    else:
        char_acc = character_accuracy(prediction, target)
    per_cell = cell_accuracy(pred_rows, target_rows)
    weighted = 0.3 * char_acc + 0.7 * per_cell
    return SampleResult(
        sample_id=sample_id,
        weighted_score=weighted,
        character_accuracy=char_acc,
        cell_accuracy=per_cell,
        error_type=classify_error(prediction, target, pred_rows, target_rows),
        segment_scores=segment_scores(pred_rows, target_rows),
    )


def aggregate(results: list[SampleResult]) -> dict[str, Any]:
    count = max(len(results), 1)
    overall = {
        "weighted_score": sum(item.weighted_score for item in results) / count,
        "character_accuracy": sum(item.character_accuracy for item in results) / count,
        "cell_accuracy": sum(item.cell_accuracy for item in results) / count,
        "by_error_type": {},
        "position_slices": {},
    }
    for result in results:
        overall["by_error_type"][result.error_type] = (
            overall["by_error_type"].get(result.error_type, 0) + 1
        )
        for key, value in result.segment_scores.items():
            overall["position_slices"][key] = overall["position_slices"].get(key, 0.0) + value
    for key in list(overall["position_slices"]):
        overall["position_slices"][key] /= count
    return overall


def evaluate(predictions_path: Path, output_path: Path | None = None) -> dict[str, Any]:
    results: list[SampleResult] = []
    with predictions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            prediction = record["prediction"]
            target = record["target_csv"]
            results.append(
                score_sample(
                    sample_id=record["sample_id"],
                    prediction=prediction,
                    target=target,
                )
            )
    payload = {
        "summary": aggregate(results),
        "samples": [
            {
                "sample_id": item.sample_id,
                "weighted_score": item.weighted_score,
                "character_accuracy": item.character_accuracy,
                "cell_accuracy": item.cell_accuracy,
                "error_type": item.error_type,
                "position_slices": item.segment_scores,
            }
            for item in results
        ],
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate raw GOT/LaTeX OCR table predictions.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = evaluate(args.predictions, args.output)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
