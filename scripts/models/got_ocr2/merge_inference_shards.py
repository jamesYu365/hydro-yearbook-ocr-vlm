#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "src"))

from yearbook_ocr.models.got_ocr2.inference import merge_prediction_shards


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge sharded GOT inference jsonl files.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Shard prediction jsonl files to merge.")
    parser.add_argument("--output", type=Path, required=True, help="Merged prediction jsonl output path.")
    args = parser.parse_args()
    output_path = merge_prediction_shards(args.inputs, args.output)
    print(output_path.as_posix())


if __name__ == "__main__":
    main()
