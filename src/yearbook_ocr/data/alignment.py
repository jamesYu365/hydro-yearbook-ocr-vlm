from __future__ import annotations

from pathlib import Path
from typing import Any

from yearbook_ocr.common.jsonl import write_json, write_jsonl
from yearbook_ocr.common.tabular import csv_rows_to_got_format, csv_rows_to_text, normalize_target_rows, parse_csv_text, read_csv_text
from yearbook_ocr.data.real_flow import parse_csv_entry, parse_image_entry, resolve_reciprocal_matches


def build_real_flow_alignment_manifest(
    dataset_dir: Path,
    image_dir: Path,
    output_path: Path,
    audit_path: Path,
) -> dict[str, Any]:
    csv_entries = [parse_csv_entry(path) for path in sorted(dataset_dir.glob("*.csv"))]
    image_entries = [parse_image_entry(path) for path in sorted(image_dir.glob("*.jpg"))]
    rows, audit = resolve_reciprocal_matches(csv_entries, image_entries)

    manifest_rows: list[dict[str, Any]] = []
    for row in rows:
        csv_text, csv_encoding = read_csv_text(Path(row["csv_path"]))
        csv_rows = parse_csv_text(csv_text)
        target_rows = normalize_target_rows(csv_rows)
        manifest_rows.append(
            {
                "sample_id": row["sample_id"],
                "station_name": row["station_name_csv"],
                "river": row["river_csv"],
                "year": row["year"],
                "csv_path": row["csv_path"],
                "csv_encoding": csv_encoding,
                "image_path": row["image_path"],
                "title_text": row["image_title_text"],
                "target_csv": csv_rows_to_text(target_rows),
                "target_got_format": csv_rows_to_got_format(target_rows),
                "match_score": row["match_score"],
                "match_status": row["match_status"],
                "split": "test",
                "source": "real",
            }
        )
    write_jsonl(output_path, manifest_rows)

    audit_payload = {
        "dataset_dir": dataset_dir.as_posix(),
        "image_dir": image_dir.as_posix(),
        **audit,
        "rows": rows,
    }
    write_json(audit_path, audit_payload)
    return audit_payload
