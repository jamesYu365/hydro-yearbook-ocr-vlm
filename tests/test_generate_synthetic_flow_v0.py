import scripts.data.generate_synthetic_flow_v0 as generator
from yearbook_ocr.common.progress import progress
from yearbook_ocr.common.tabular import seeded_random


def test_resolve_font_path_prefers_first_existing_candidate(tmp_path, monkeypatch) -> None:
    missing = tmp_path / "missing.ttf"
    chosen = tmp_path / "chosen.ttf"
    chosen.write_text("not a real font", encoding="utf-8")
    later = tmp_path / "later.ttf"
    later.write_text("not a real font", encoding="utf-8")

    monkeypatch.setattr(generator, "DEFAULT_FONT_CANDIDATES", (missing, chosen, later))

    assert generator.resolve_font_path(None) == chosen


def test_resolve_font_path_respects_explicit_font(tmp_path) -> None:
    explicit = tmp_path / "explicit.ttf"
    explicit.write_text("not a real font", encoding="utf-8")

    assert generator.resolve_font_path(explicit) == explicit


def test_table_grid_lines_keep_only_top_header_and_column_lines() -> None:
    lines = generator.table_grid_lines(
        row_count=4,
        col_count=3,
        left=10,
        top=20,
        col_width=5,
        header_height=9,
        row_height=7,
    )

    assert (10, 20, 25, 20) in lines
    assert (10, 29, 25, 29) in lines
    assert (10, 20, 10, 50) in lines
    assert (15, 20, 15, 50) in lines
    assert (20, 20, 20, 50) in lines
    assert (25, 20, 25, 50) in lines

    assert (25, 50, 10, 50) not in lines
    assert (10, 34, 15, 34) not in lines
    assert (10, 41, 15, 41) not in lines
    assert (15, 34, 25, 34) not in lines
    assert (15, 41, 25, 41) not in lines


def test_render_table_uses_real_crop_like_default_size() -> None:
    rows = [
        [r"日\月", "一月", "二月"],
        ["1", "10", "20"],
        ["", "", ""],
        ["2", "11", "21"],
    ]
    font_path = generator.resolve_font_path(None)

    image, cells = generator.render_table(rows, font_path, seeded_random(1))

    assert image.size == (
        generator.DEFAULT_RENDER_CONFIG.image_width,
        generator.DEFAULT_RENDER_CONFIG.image_height,
    )
    assert image.size == (2160, 1090)
    assert cells[0]["bbox"] == [60, 35, 740, 105]
    assert cells[3]["bbox"] == [60, 105, 740, 130]
    assert len(cells) == 12


def test_default_render_config_matches_real_crop_density() -> None:
    config = generator.DEFAULT_RENDER_CONFIG

    assert config.image_width == 2160
    assert config.image_height == 1090
    assert config.margin_x == 60
    assert config.margin_y == 35
    assert config.header_height == 70
    assert config.row_height == 25
    assert config.body_font_size == 17


def test_normalize_table_header_uses_day_month_label() -> None:
    rows = generator.normalize_table_header(
        [
            ["日期", "一月", "二月"],
            ["1", "10", "20"],
            ["", "", ""],
        ]
    )

    assert rows[0][0] == r"日\月"
    assert rows[2] == ["", "", ""]


def test_sample_table_rows_preserves_blank_separator_rows() -> None:
    template_rows = [
        ["日期", "一月", "二月"],
        ["1", "10", "20"],
        ["", "", ""],
        ["2", "11", "21"],
    ]
    pools = {month_index: [str(100 + month_index)] for month_index in range(12)}

    rows = generator.sample_table_rows(template_rows, pools, seeded_random(1))

    assert rows[2] == ["", "", ""]


def test_progress_accepts_total() -> None:
    assert list(progress([1, 2], total=2, disable=True)) == [1, 2]


def test_manifest_record_can_be_rebuilt_from_existing_assets(tmp_path) -> None:
    template_rows = [
        ["日期", "一月", "二月"],
        ["1", "10", "20"],
        ["", "", ""],
        ["2", "11", "21"],
    ]
    pools = {month_index: [str(100 + month_index)] for month_index in range(12)}
    image_dir = tmp_path / "images"
    layout_dir = tmp_path / "layouts"
    image_dir.mkdir()
    layout_dir.mkdir()
    (image_dir / "flow_v0_00000.png").write_bytes(b"fake")
    (layout_dir / "flow_v0_00000.json").write_text(
        '{"perturbations":[{"type":"brightness","strength":0.1}]}',
        encoding="utf-8",
    )

    split, record = generator.build_manifest_record_from_existing_assets(
        0,
        20260408,
        "val",
        template_rows,
        pools,
        image_dir,
        layout_dir,
    )

    assert split == "val"
    assert record["sample_id"] == generator.sample_id_from_name("flow_v0_00000")
    assert record["target_csv"].startswith(r"日\月,一月,二月")
    assert "\n,,\n" not in record["target_csv"]
    assert record["target_got_format"].startswith("\\begin{tabular}")
    body_lines = [line for line in record["target_got_format"].splitlines() if line.endswith("\\\\")]
    assert all(any(cell.strip() for cell in line.removesuffix("\\\\").split("&")) for line in body_lines)
    assert record["perturbations"] == [{"type": "brightness", "strength": 0.1}]


def test_build_split_map_uses_exact_shuffled_val_count() -> None:
    split_map = generator.build_split_map(num_samples=10000, val_ratio=0.2, seed=20260408)

    assert len(split_map) == 10000
    assert sum(1 for split in split_map.values() if split == "val") == 2000
    assert sum(1 for split in split_map.values() if split == "train") == 8000
    assert split_map == generator.build_split_map(num_samples=10000, val_ratio=0.2, seed=20260408)
