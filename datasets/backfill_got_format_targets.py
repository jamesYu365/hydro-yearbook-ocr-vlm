#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.common.yearbook_flow_common import csv_rows_to_got_format, parse_csv_text


def backfill_manifest(path: Path) -> int:
    rows = []
    updated = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if "target_csv" in record and "target_got_format" not in record:
                record["target_got_format"] = csv_rows_to_got_format(parse_csv_text(record["target_csv"]))
                updated += 1
            rows.append(record)
    with path.open("w", encoding="utf-8") as handle:
        for record in rows:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill target_got_format into manifest files.")
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        updated = backfill_manifest(path)
        print(json.dumps({"path": path.as_posix(), "updated": updated}, ensure_ascii=False))


if __name__ == "__main__":
    main()
