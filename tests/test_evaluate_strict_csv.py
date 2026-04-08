from scripts.eval.evaluate_strict_csv import score_sample


def test_value_error_keeps_structure() -> None:
    target = "日期,一月\n1,12\n2,13\n"
    prediction = "日期,一月\n1,12\n2,99\n"
    result = score_sample("sample_1", prediction, target)
    assert result.error_type == "value_error"
    assert result.cell_accuracy == 0.75


def test_truncation_is_detected() -> None:
    target = "日期,一月\n1,12\n2,13\n"
    prediction = "日期,一月\n1,12\n"
    result = score_sample("sample_2", prediction, target)
    assert result.error_type == "truncation"


def test_structure_error_zeroes_cell_accuracy() -> None:
    target = "日期,一月\n1,12\n2,13\n"
    prediction = "日期,一月\n1\n2,13\n"
    result = score_sample("sample_3", prediction, target)
    assert result.error_type == "structure_error"
    assert result.cell_accuracy == 0.0
