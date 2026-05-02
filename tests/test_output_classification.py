from __future__ import annotations

import json
from pathlib import Path

from yearbook_ocr.eval.output_classification import classify_outputs


def latex_table(rows: list[list[str]]) -> str:
    body = "\n".join(" & ".join(row) + r" \\" for row in rows)
    return "\\begin{tabular}{|c|c|}\n" + body + "\n\\end{tabular}\n"


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_classify_outputs_copies_raw_files_into_correctness_folders(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    raw_dir = tmp_path / "per_image_raw"
    output_dir = tmp_path / "classified"
    raw_dir.mkdir()

    write_jsonl(
        predictions,
        [
            {
                "sample_id": "perfect",
                "image_path": "images/perfect.jpg",
                "prediction": latex_table([["日期", "一月"], ["1", "12"]]),
                "target_csv": "日期,一月\n1,12\n",
            },
            {
                "sample_id": "structure_bad",
                "image_path": "images/structure_bad.jpg",
                "prediction": "日\\月\n1\n",
                "target_csv": "日期,一月\n1,12\n",
            },
        ],
    )
    (raw_dir / "perfect.txt").write_text("perfect raw\n", encoding="utf-8")
    (raw_dir / "structure_bad.txt").write_text("bad raw\n", encoding="utf-8")

    summary = classify_outputs(predictions, raw_dir=raw_dir, output_dir=output_dir)

    assert summary["folder_counts"]["structure_correct__cell_correct"] == 1
    assert summary["folder_counts"]["structure_error__cell_error"] == 1
    assert (output_dir / "structure_correct__cell_correct" / "perfect.txt").read_text(encoding="utf-8") == "perfect raw\n"
    assert (output_dir / "structure_error__cell_error" / "structure_bad.txt").read_text(encoding="utf-8") == "bad raw\n"
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "samples.jsonl").exists()
