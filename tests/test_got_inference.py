from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from yearbook_ocr.models.got_ocr2.inference import (
    build_child_command,
    default_eval_dir,
    default_output_path,
    default_per_image_dir,
    format_per_image_prediction,
    index_existing_records,
    merge_prediction_shards,
    merge_records_preserving_existing,
    parse_gpu_ids,
    per_image_extension,
    per_image_file_stem,
    pretty_latex_tabular,
    remove_shard_outputs,
    with_shard_suffix,
)


def make_args(**overrides) -> argparse.Namespace:
    payload = {
        "manifest": Path("data/manifests/flow_real_test_aligned.jsonl"),
        "backend": "official_chat",
        "query_mode": "official_format",
        "dtype": "float16",
        "max_new_tokens": 4096,
        "cache_root": Path("outputs/cache"),
        "base_model": Path("outputs/cache/modelscope/models/stepfun-ai/GOT-OCR2_0"),
        "adapter_dir": None,
        "query": None,
        "limit": None,
        "per_image_format": None,
        "per_image_dir": None,
        "overwrite": False,
    }
    payload.update(overrides)
    return argparse.Namespace(**payload)


def test_parse_gpu_ids_keeps_stable_order() -> None:
    assert parse_gpu_ids("1, 2,3 , 7") == ["1", "2", "3", "7"]


def test_parse_gpu_ids_rejects_empty_string() -> None:
    with pytest.raises(ValueError, match="--gpu-ids must contain at least one GPU id."):
        parse_gpu_ids(" , ")


def test_with_shard_suffix_preserves_extension() -> None:
    result = with_shard_suffix(Path("outputs/got_ocr2_base/result.jsonl"), shard_id=2, num_shards=7)
    assert result == Path("outputs/got_ocr2_base/result.shard2of7.jsonl")


def test_build_child_command_includes_shard_and_optional_args() -> None:
    args = make_args(
        adapter_dir=Path("outputs/run/checkpoint-90"),
        query="OCR with format: ",
        limit=35,
        per_image_format="latex",
        per_image_dir=Path("outputs/run/per_image_latex"),
    )
    command = build_child_command(
        args=args,
        shard_id=3,
        num_shards=7,
        shard_output=Path("/tmp/shard3.jsonl"),
    )
    assert "--num-shards" in command
    assert "--shard-id" in command
    assert "--adapter-dir" in command
    assert "--query" in command
    assert "--limit" in command
    assert "--per-image-format" in command
    assert "--per-image-dir" in command
    assert "--overwrite" not in command
    assert command[command.index("--shard-id") + 1] == "3"
    assert command[command.index("--num-shards") + 1] == "7"
    assert command[command.index("--output") + 1] == "/tmp/shard3.jsonl"


def test_build_child_command_forwards_overwrite() -> None:
    args = make_args(overwrite=True)
    command = build_child_command(
        args=args,
        shard_id=0,
        num_shards=2,
        shard_output=Path("/tmp/shard0.jsonl"),
    )

    assert "--overwrite" in command


def test_default_per_image_dir_derives_from_output_name() -> None:
    result = default_per_image_dir(Path("outputs/got_ocr2_base/flow_real_first5_official_chat.jsonl"), "raw")
    assert result == Path("outputs/got_ocr2_base/per_image_raw")


def test_default_per_image_dir_uses_simple_format_subdir() -> None:
    result = default_per_image_dir(
        Path("outputs/got_ocr2_v1_swift/v2/eval/checkpoint-90/flow_real_all_official_chat.jsonl"),
        "latex",
    )
    assert result == Path("outputs/got_ocr2_v1_swift/v2/eval/checkpoint-90/per_image_latex")


def test_per_image_extension_uses_semantic_file_suffixes() -> None:
    assert per_image_extension("raw") == "txt"
    assert per_image_extension("latex") == "tex"


def test_per_image_file_stem_prefers_image_path_stem() -> None:
    assert per_image_file_stem(
        {
            "sample_id": "flow_4f99d5a57398",
            "image_path": "datasets/derived/page_0002_table0_汉江仙桃(二)站_2006.jpg",
        }
    ) == "page_0002_table0_汉江仙桃(二)站_2006"


def test_per_image_file_stem_falls_back_to_sample_id() -> None:
    assert per_image_file_stem({"sample_id": "flow_4f99d5a57398"}) == "flow_4f99d5a57398"


def test_pretty_latex_tabular_splits_hline_from_rows() -> None:
    text = (
        "\\begin{tabular}{|c|c|}\n"
        "\\hline 日期 & 一月 \\\\\n"
        "\\hline 1 & 12 \\\\\n"
        "\\hline\\end{tabular}\n"
    )

    assert pretty_latex_tabular(text) == (
        "\\begin{tabular}{|c|c|}\n"
        "\\hline\n"
        "日期 & 一月 \\\\\n"
        "\\hline\n"
        "1 & 12 \\\\\n"
        "\\hline\n"
        "\\end{tabular}\n"
    )


def test_format_per_image_prediction_keeps_raw_text_unmodified_except_final_newline() -> None:
    assert format_per_image_prediction("a\r\nb", "raw") == "a\nb\n"


def test_index_existing_records_loads_jsonl_by_sample_id(tmp_path: Path) -> None:
    output = tmp_path / "predictions.jsonl"
    output.write_text(
        '{"sample_id":"a","prediction":"old-a"}\n'
        '{"sample_id":"b","prediction":"old-b"}\n',
        encoding="utf-8",
    )

    records = index_existing_records(output, overwrite=False)

    assert sorted(records) == ["a", "b"]
    assert records["a"]["prediction"] == "old-a"


def test_index_existing_records_ignores_existing_when_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "predictions.jsonl"
    output.write_text('{"sample_id":"a","prediction":"old-a"}\n', encoding="utf-8")

    assert index_existing_records(output, overwrite=True) == {}


def test_index_existing_records_ignores_missing_output(tmp_path: Path) -> None:
    assert index_existing_records(tmp_path / "missing.jsonl", overwrite=False) == {}


def test_merge_records_preserving_existing_keeps_unselected_records() -> None:
    existing = {
        "a": {"sample_id": "a", "prediction": "old-a"},
        "b": {"sample_id": "b", "prediction": "old-b"},
    }
    selected = [{"sample_id": "a", "prediction": "new-a"}]

    assert merge_records_preserving_existing(existing, selected) == [
        {"sample_id": "a", "prediction": "new-a"},
        {"sample_id": "b", "prediction": "old-b"},
    ]


def test_merge_records_preserving_existing_appends_new_records() -> None:
    existing = {"a": {"sample_id": "a", "prediction": "old-a"}}
    selected = [{"sample_id": "b", "prediction": "new-b"}]

    assert merge_records_preserving_existing(existing, selected) == [
        {"sample_id": "a", "prediction": "old-a"},
        {"sample_id": "b", "prediction": "new-b"},
    ]


def test_default_eval_dir_for_adapter_checkpoint() -> None:
    result = default_eval_dir(Path("outputs/got_ocr2_v1_swift/v2-20260424-095909/checkpoint-90"))
    assert result == Path("outputs/got_ocr2_v1_swift/v2-20260424-095909/eval/checkpoint-90")


def test_default_eval_dir_for_base_model() -> None:
    assert default_eval_dir(None) == Path("outputs/got_ocr2_base/eval/checkpoint-0")


def test_default_output_path_uses_checkpoint_eval_dir() -> None:
    result = default_output_path(
        adapter_dir=Path("outputs/got_ocr2_v1_swift/v2-20260424-095909/checkpoint-90"),
        backend="official_chat",
        single_image=False,
        image_path=None,
        limit=None,
        shard_id=None,
        num_shards=None,
    )
    assert result == Path(
        "outputs/got_ocr2_v1_swift/v2-20260424-095909/eval/checkpoint-90/flow_real_all_official_chat.jsonl"
    )


def test_default_output_path_uses_base_checkpoint_zero() -> None:
    result = default_output_path(
        adapter_dir=None,
        backend="official_chat",
        single_image=False,
        image_path=None,
        limit=5,
        shard_id=None,
        num_shards=None,
    )
    assert result == Path("outputs/got_ocr2_base/eval/checkpoint-0/flow_real_first5_official_chat.jsonl")


def test_remove_shard_outputs_deletes_merged_shards(tmp_path: Path) -> None:
    shard0 = tmp_path / "pred.shard0of2.jsonl"
    shard1 = tmp_path / "pred.shard1of2.jsonl"
    output = tmp_path / "pred.jsonl"
    shard0.write_text('{"sample_id":"b","prediction":"2"}\n', encoding="utf-8")
    shard1.write_text('{"sample_id":"a","prediction":"1"}\n', encoding="utf-8")

    merge_prediction_shards([shard0, shard1], output)
    remove_shard_outputs([shard0, shard1])

    assert output.exists()
    assert not shard0.exists()
    assert not shard1.exists()
