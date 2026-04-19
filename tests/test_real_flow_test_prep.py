from pathlib import Path

from datasets.real_flow_test_prep import (
    CsvEntry,
    ImageEntry,
    OcrToken,
    apply_bottom_buffer,
    fallback_cut_y_from_lines,
    find_horizontal_cut_y,
    normalize_text,
    resolve_reciprocal_matches,
    score_match,
    select_statistics_anchor,
)


def test_normalize_text_handles_common_ocr_drift() -> None:
    assert normalize_text("水環潭(电站)站") == "涢水澴潭(电站)"
    assert normalize_text("澴水不花园站") == "澴水花园"
    assert normalize_text("大富水应城(二)站") == "大富水应城"


def test_score_match_prefers_expected_image() -> None:
    csv_entry = CsvEntry(
        sample_id="flow_x",
        csv_path=Path("datasets/流量/2006/花园_2006_澴水_已校准.csv"),
        station_name="花园",
        river="澴水",
        year="2006",
    )
    best = ImageEntry(
        image_path=Path("page_0018_table0_澴水不花园站_2006.jpg"),
        stable_name="page_0018_table0",
        title_text="澴水不花园站",
        year="2006",
    )
    wrong = ImageEntry(
        image_path=Path("page_0017_table1_澴水草店站_2006.jpg"),
        stable_name="page_0017_table1",
        title_text="澴水草店站",
        year="2006",
    )
    assert score_match(csv_entry, best).score > score_match(csv_entry, wrong).score


def test_resolve_reciprocal_matches_marks_clear_pairs_confirmed() -> None:
    csv_entries = [
        CsvEntry("a", Path("皇庄.csv"), "皇庄", "汉江", "2006"),
        CsvEntry("b", Path("草店.csv"), "草店", "澴水", "2006"),
    ]
    image_entries = [
        ImageEntry(Path("page_0001_table0_汉江皇庄站_2006.jpg"), "page_0001_table0", "汉江皇庄站", "2006"),
        ImageEntry(Path("page_0017_table1_澴水草店站_2006.jpg"), "page_0017_table1", "澴水草店站", "2006"),
    ]
    rows, audit = resolve_reciprocal_matches(csv_entries, image_entries, min_score=60.0, min_csv_margin=5.0, min_image_margin=5.0)
    assert audit["summary"]["confirmed_count"] == 2
    assert {row["match_status"] for row in rows} == {"confirmed"}


def test_select_statistics_anchor_uses_upper_average_not_year_stats_neighbor() -> None:
    tokens = [
        OcrToken("平均", 0, 100, 10, 120),
        OcrToken("平均", 0, 160, 10, 180),
        OcrToken("年统计", 0, 158, 20, 182),
    ]
    anchor, method = select_statistics_anchor(tokens)
    assert anchor is not None
    assert anchor.top == 100
    assert method == "average_anchor"


def test_select_statistics_anchor_rejects_year_stats_only_detection() -> None:
    tokens = [
        OcrToken("年统计", 0, 158, 20, 182),
    ]
    anchor, method = select_statistics_anchor(tokens)
    assert anchor is None
    assert method is None


def test_select_statistics_anchor_accepts_single_average_above_year_stats() -> None:
    tokens = [
        OcrToken("平均", 0, 100, 10, 120),
        OcrToken("年统计", 0, 158, 20, 182),
    ]
    anchor, method = select_statistics_anchor(tokens)
    assert anchor is not None
    assert anchor.top == 100
    assert method == "average_anchor"


def test_select_statistics_anchor_accepts_weak_ping_anchor_above_year_stats() -> None:
    tokens = [
        OcrToken("平", 44, 126, 76, 158),
        OcrToken("日期", 35, 181, 141, 229),
        OcrToken("最低", 38, 214, 140, 256),
        OcrToken("年统计", 37, 275, 139, 317),
    ]
    anchor, method = select_statistics_anchor(tokens)
    assert anchor is not None
    assert anchor.text == "平"
    assert method == "weak_average_anchor"


def test_select_statistics_anchor_accepts_weak_jun_anchor_above_year_stats() -> None:
    tokens = [
        OcrToken("均", 78, 126, 110, 158),
        OcrToken("日期", 35, 181, 141, 229),
        OcrToken("最低", 38, 214, 140, 256),
        OcrToken("年统计", 37, 275, 139, 317),
    ]
    anchor, method = select_statistics_anchor(tokens)
    assert anchor is not None
    assert anchor.text == "均"
    assert method == "weak_average_anchor"


def test_select_statistics_anchor_rejects_weak_anchor_without_statistics_context() -> None:
    tokens = [
        OcrToken("平", 44, 126, 76, 158),
        OcrToken("年统计", 37, 275, 139, 317),
    ]
    anchor, method = select_statistics_anchor(tokens)
    assert anchor is None
    assert method is None


def test_select_statistics_anchor_rejects_weak_anchor_in_value_column() -> None:
    tokens = [
        OcrToken("平", 240, 126, 272, 158),
        OcrToken("日期", 35, 181, 141, 229),
        OcrToken("最低", 38, 214, 140, 256),
        OcrToken("年统计", 37, 275, 139, 317),
    ]
    anchor, method = select_statistics_anchor(tokens)
    assert anchor is None
    assert method is None


def test_find_horizontal_cut_y_prefers_line_above_anchor() -> None:
    row_dark_counts = [0] * 220
    row_dark_counts[140] = 120
    row_dark_counts[180] = 120
    cut_y = find_horizontal_cut_y(row_dark_counts, width=180, anchor_top=150, min_ratio=0.6, search_up_px=20)
    assert cut_y == 141


def test_fallback_cut_y_uses_first_strong_line_in_lower_region() -> None:
    row_dark_counts = [0] * 300
    row_dark_counts[210] = 150
    row_dark_counts[240] = 150
    cut_y = fallback_cut_y_from_lines(row_dark_counts, width=200, height=300, min_ratio=0.6, start_ratio=0.6)
    assert cut_y == 211


def test_apply_bottom_buffer_expands_cut_without_exceeding_height() -> None:
    assert apply_bottom_buffer(120, height=200, bottom_buffer_px=6) == 126
    assert apply_bottom_buffer(198, height=200, bottom_buffer_px=6) == 200
