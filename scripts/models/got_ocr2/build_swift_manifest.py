from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "src"))

from yearbook_ocr.common.tabular import DEFAULT_GOT_FORMAT_PROMPT
from yearbook_ocr.data.manifests import build_swift_records


def convert_records(input_path: Path, output_path: Path, prompt: str, response_field: str) -> int:
    return build_swift_records(input_path, output_path, prompt, response_field)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ms-swift manifest for GOT-OCR2.0.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_GOT_FORMAT_PROMPT,
        help="Prompt to prepend after <image>. Defaults to the official GOT format prompt.",
    )
    parser.add_argument(
        "--response-field",
        type=str,
        default="target_got_format",
        help="Manifest field to use as the training response. Defaults to target_got_format.",
    )
    args = parser.parse_args()
    count = convert_records(args.input, args.output, args.prompt, args.response_field)
    print(f"Wrote {count} ms-swift records to {args.output}")


if __name__ == "__main__":
    main()
