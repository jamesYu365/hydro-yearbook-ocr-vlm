from pathlib import Path

from scripts.common.yearbook_flow_common import (
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
