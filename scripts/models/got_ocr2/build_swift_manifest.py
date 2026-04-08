from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from scripts.common.yearbook_flow_common import DEFAULT_PROMPT


def convert_records(input_path: Path, output_path: Path, prompt: str) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with input_path.open("r", encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            if not line.strip():
                continue
            record = json.loads(line)
            payload = {
                "query": f"<image>{prompt}",
                "response": record["target_csv"],
                "images": [record["image_path"]],
                "sample_id": record["sample_id"],
            }
            dst.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ms-swift manifest for GOT-OCR2.0.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    args = parser.parse_args()
    count = convert_records(args.input, args.output, args.prompt)
    print(f"Wrote {count} ms-swift records to {args.output}")


if __name__ == "__main__":
    main()

