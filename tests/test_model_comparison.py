from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from yearbook_ocr.eval.model_comparison import (
    compare_predictions,
    index_records,
    parse_latex_tabular,
    score_structure_cell,
)


def latex_table(rows: list[list[str]]) -> str:
    body = "\n".join(" & ".join(row) + r" \\" for row in rows)
    return "\\begin{tabular}{|c|c|}\n" + body + "\n\\end{tabular}\n"


def record(sample_id: str, prediction: str, target: str = "日期,一月\n1,12\n2,13\n") -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "image_path": f"images/{sample_id}.jpg",
        "prediction": prediction,
        "target_csv": target,
    }


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_score_structure_cell_all_correct() -> None:
    result = score_structure_cell(record("sample_1", latex_table([["日期", "一月"], ["1", "12"], ["2", "13"]])))

    assert result.structure_correct is True
    assert result.cell_accuracy == 1.0
    assert result.cell_correct_count == 6
    assert result.total_cell_count == 6


def test_parse_latex_tabular_handles_got_format_rows() -> None:
    rows = parse_latex_tabular(
        "\\begin{tabular}{|c|c|}\n"
        "\\hline 日期 & 一月 \\\\\n"
        "\\hline 1 & 12 \\\\\n"
        "\\hline  &  \\\\\n"
        "\\hline\n"
        "\\end{tabular}\n"
    )

    assert rows == [["日期", "一月"], ["1", "12"], ["", ""]]


def test_score_structure_cell_accepts_latex_prediction() -> None:
    prediction = (
        "\\begin{tabular}{|c|c|}\n"
        "\\hline 日期 & 一月 \\\\\n"
        "\\hline 1 & 12 \\\\\n"
        "\\hline 2 & 13 \\\\\n"
        "\\hline\n"
        "\\end{tabular}\n"
    )
    result = score_structure_cell(record("sample_1", prediction))

    assert result.structure_correct is True
    assert result.cell_accuracy == 1.0


def test_score_structure_cell_value_error_keeps_structure() -> None:
    result = score_structure_cell(record("sample_1", latex_table([["日期", "一月"], ["1", "12"], ["2", "99"]])))

    assert result.structure_correct is True
    assert result.cell_accuracy == 5 / 6
    assert result.error_type == "value_error"


def test_score_structure_cell_row_count_mismatch_zeroes_cells() -> None:
    result = score_structure_cell(record("sample_1", latex_table([["日期", "一月"], ["1", "12"]])))

    assert result.structure_correct is False
    assert result.cell_accuracy == 0.0
    assert result.error_type == "truncation"


def test_score_structure_cell_column_count_mismatch_zeroes_cells() -> None:
    result = score_structure_cell(record("sample_1", "\\begin{tabular}{|c|c|}\n日期 & 一月 \\\\\n1 \\\\\n2 & 13 \\\\\n\\end{tabular}\n"))

    assert result.structure_correct is False
    assert result.cell_accuracy == 0.0
    assert result.error_type == "structure_error"


def test_score_structure_cell_unparsable_prediction() -> None:
    result = score_structure_cell(record("sample_1", "日\\月\n1\n2\n"))

    assert result.structure_correct is False
    assert result.cell_accuracy == 0.0
    assert result.error_type == "structure_error"


def test_index_records_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="duplicate sample_id"):
        index_records(
            [
                record("x", latex_table([["日期", "一月"], ["1", "12"]])),
                record("x", latex_table([["日期", "一月"], ["1", "12"]])),
            ],
            "before",
        )


def test_compare_predictions_writes_summary_and_transitions(tmp_path: Path) -> None:
    before_path = tmp_path / "before.jsonl"
    after_path = tmp_path / "after.jsonl"
    output_dir = tmp_path / "comparison"
    write_jsonl(
        before_path,
            [
                record("improves", latex_table([["日期", "一月"], ["1"], ["2", "13"]])),
                record("regresses", latex_table([["日期", "一月"], ["1", "12"], ["2", "13"]])),
                record("ties", latex_table([["日期", "一月"], ["1", "12"], ["2", "99"]])),
            ],
        )
    write_jsonl(
        after_path,
            [
                record("improves", latex_table([["日期", "一月"], ["1", "12"], ["2", "13"]])),
                record("regresses", latex_table([["日期", "一月"], ["1"], ["2", "13"]])),
                record("ties", latex_table([["日期", "一月"], ["1", "12"], ["2", "99"]])),
            ],
        )

    payload = compare_predictions(before_path, after_path, "base", "ckpt", output_dir, write_images=False)

    assert payload["before"]["sample_count"] == 3
    assert payload["after"]["sample_count"] == 3
    assert payload["before"]["structure_accuracy"] == 2 / 3
    assert payload["after"]["structure_accuracy"] == 2 / 3
    assert payload["delta"]["structure_correct_count"] == 0
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "summary.csv").exists()
    assert (output_dir / "samples.csv").exists()
    assert (output_dir / "structure_transitions.csv").exists()

    with (output_dir / "structure_transitions.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = {row["transition"]: int(row["count"]) for row in csv.DictReader(handle)}
    assert rows == {
        "correct -> correct": 1,
        "correct -> incorrect": 1,
        "incorrect -> correct": 1,
    }
