from pathlib import Path

from yearbook_ocr.data.manifests import backfill_got_format_manifest


def test_backfill_manifest_adds_target_got_format(tmp_path: Path) -> None:
    manifest = tmp_path / "samples.jsonl"
    manifest.write_text(
        '{"sample_id":"x","target_csv":"日期,一月\\n1,29.5\\n"}\n',
        encoding="utf-8",
    )
    updated = backfill_got_format_manifest(manifest)
    content = manifest.read_text(encoding="utf-8")
    assert updated == 1
    assert '"target_got_format"' in content
    assert "\\\\begin{tabular}" in content
