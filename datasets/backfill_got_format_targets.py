#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from yearbook_ocr.data.manifests import backfill_got_format_manifest


def backfill_manifest(path: Path) -> int:
    return backfill_got_format_manifest(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill target_got_format into manifest files.")
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        updated = backfill_manifest(path)
        print(json.dumps({"path": path.as_posix(), "updated": updated}, ensure_ascii=False))


if __name__ == "__main__":
    main()
