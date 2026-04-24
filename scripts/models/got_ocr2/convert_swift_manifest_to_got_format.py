#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "src"))

from yearbook_ocr.common.tabular import DEFAULT_GOT_FORMAT_PROMPT
from yearbook_ocr.data.manifests import convert_swift_manifest_to_got_format as convert_manifest_impl


def convert_manifest(path: Path, prompt: str) -> int:
    return convert_manifest_impl(path, prompt)


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
