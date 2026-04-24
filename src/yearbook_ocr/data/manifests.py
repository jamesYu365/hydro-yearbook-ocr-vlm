from __future__ import annotations

from pathlib import Path

from yearbook_ocr.common.jsonl import load_jsonl, write_jsonl
from yearbook_ocr.common.tabular import DEFAULT_GOT_FORMAT_PROMPT, csv_rows_to_got_format, parse_csv_text


def backfill_got_format_manifest(path: Path) -> int:
    rows = load_jsonl(path)
    updated = 0
    for record in rows:
        if "target_csv" in record and "target_got_format" not in record:
            record["target_got_format"] = csv_rows_to_got_format(parse_csv_text(record["target_csv"]))
            updated += 1
    write_jsonl(path, rows)
    return updated


def convert_swift_manifest_to_got_format(path: Path, prompt: str = DEFAULT_GOT_FORMAT_PROMPT) -> int:
    rows = load_jsonl(path)
    updated = 0
    for record in rows:
        csv_text = record.get("response")
        if csv_text is None:
            raise KeyError(f"Missing response in {path}")
        record["query"] = f"<image>{prompt}"
        record["response"] = csv_rows_to_got_format(parse_csv_text(csv_text))
        updated += 1
    write_jsonl(path, rows)
    return updated


def build_swift_records(
    input_path: Path,
    output_path: Path,
    prompt: str,
    response_field: str,
) -> int:
    source_rows = load_jsonl(input_path)
    payload_rows: list[dict[str, object]] = []
    for record in source_rows:
        if response_field not in record:
            raise KeyError(f"Missing response field '{response_field}' in {input_path}")
        payload_rows.append(
            {
                "query": f"<image>{prompt}",
                "response": record[response_field],
                "images": [record["image_path"]],
                "sample_id": record["sample_id"],
            }
        )
    write_jsonl(output_path, payload_rows)
    return len(payload_rows)
