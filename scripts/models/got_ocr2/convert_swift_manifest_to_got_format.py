#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from scripts.common.yearbook_flow_common import DEFAULT_GOT_FORMAT_PROMPT, csv_rows_to_got_format, parse_csv_text


def convert_manifest(path: Path, prompt: str) -> int:
    rows = []
    updated = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            csv_text = record.get("response")
            if csv_text is None:
                raise KeyError(f"Missing response in {path}")
            record["query"] = f"<image>{prompt}"
            record["response"] = csv_rows_to_got_format(parse_csv_text(csv_text))
            updated += 1
            rows.append(record)
    with path.open("w", encoding="utf-8") as handle:
        for record in rows:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert existing swift manifests from CSV targets to GOT format.")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--prompt", type=str, default=DEFAULT_GOT_FORMAT_PROMPT)
    args = parser.parse_args()
    for path in args.paths:
        updated = convert_manifest(path, args.prompt)
        print(json.dumps({"path": path.as_posix(), "updated": updated}, ensure_ascii=False))


if __name__ == "__main__":
    main()
