import json

from scripts.eval.evaluate_predictions import evaluate, score_sample


def latex_table(rows: list[list[str]]) -> str:
    body = "\n".join(" & ".join(row) + r" \\" for row in rows)
    return "\\begin{tabular}{|c|c|}\n" + body + "\n\\end{tabular}\n"


def test_value_error_keeps_structure() -> None:
    target = "日期,一月\n1,12\n2,13\n"
    prediction = latex_table([["日期", "一月"], ["1", "12"], ["2", "99"]])
    result = score_sample("sample_1", prediction, target)
    assert result.error_type == "value_error"
    assert result.cell_accuracy == 5 / 6


def test_truncation_is_detected() -> None:
    target = "日期,一月\n1,12\n2,13\n"
    prediction = latex_table([["日期", "一月"], ["1", "12"]])
    result = score_sample("sample_2", prediction, target)
    assert result.error_type == "truncation"


def test_structure_error_zeroes_cell_accuracy() -> None:
    target = "日期,一月\n1,12\n2,13\n"
    prediction = "\\begin{tabular}{|c|c|}\n日期 & 一月 \\\\\n1 \\\\\n2 & 13 \\\\\n\\end{tabular}\n"
    result = score_sample("sample_3", prediction, target)
    assert result.error_type == "structure_error"
    assert result.cell_accuracy == 0.0


def test_got_latex_prediction_scores_against_csv_target() -> None:
    target = "日期,一月\n1,12\n2,13\n"
    prediction = (
        "\\begin{tabular}{|c|c|}\n"
        "\\hline\n"
        "日期 & 一月 \\\\\n"
        "\\hline\n"
        "1 & 12 \\\\\n"
        "\\hline\n"
        "2 & 13 \\\\\n"
        "\\hline\n"
        "\\end{tabular}\n"
    )
    result = score_sample("sample_4", prediction, target)
    assert result.error_type == "value_error"
    assert result.cell_accuracy == 1.0
    assert result.character_accuracy == 1.0


def test_unparseable_raw_output_is_structure_error(tmp_path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    target = "日期,一月\n1,12\n"
    prediction = "日\\月\n1\n2\n"
    predictions.write_text(
        json.dumps(
            {
                "sample_id": "sample_5",
                "prediction": prediction,
                "target_csv": target,
                "target_got_format": prediction,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    payload = evaluate(predictions)

    assert payload["summary"]["cell_accuracy"] == 0.0
    assert payload["summary"]["by_error_type"] == {"structure_error": 1}
