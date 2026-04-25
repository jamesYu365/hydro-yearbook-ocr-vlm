from pathlib import Path

from yearbook_ocr.common.tabular import (
    csv_rows_to_got_format,
    remove_blank_rows,
    month_day_limit,
    parse_station_meta,
    sample_id_from_name,
)


def test_parse_station_meta() -> None:
    meta = parse_station_meta(Path("皇庄_2006_汉江_已校准.csv"))
    assert meta.station_name == "皇庄"
    assert meta.year == "2006"
    assert meta.river == "汉江"


def test_month_day_limit() -> None:
    assert month_day_limit(0) == 31
    assert month_day_limit(1) == 28
    assert month_day_limit(3) == 30


def test_sample_id_is_stable() -> None:
    assert sample_id_from_name("flow_v0_00001") == sample_id_from_name("flow_v0_00001")


def test_csv_rows_to_got_format() -> None:
    rows = [
        ["日期", "一月", "二月"],
        ["1", "29.5", "19"],
        ["2", "", "19.2"],
    ]
    text = csv_rows_to_got_format(rows)
    assert text.startswith("\\begin{tabular}{|c|c|c|}\n\\hline\n")
    assert "日期 & 一月 & 二月 \\\\" in text
    assert "1 & 29.5 & 19 \\\\" in text
    assert "2 &  & 19.2 \\\\" in text
    assert text.endswith("\\end{tabular}\n")


def test_remove_blank_rows_keeps_empty_cells_inside_data_rows() -> None:
    rows = [
        ["日期", "一月", "二月"],
        ["1", "29.5", ""],
        ["", "", ""],
        ["2", "", "19.2"],
    ]

    assert remove_blank_rows(rows) == [
        ["日期", "一月", "二月"],
        ["1", "29.5", ""],
        ["2", "", "19.2"],
    ]
