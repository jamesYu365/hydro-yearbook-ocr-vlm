#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from yearbook_ocr.data.alignment import build_real_flow_alignment_manifest


DEFAULT_IMAGE_DIR = Path(
    "datasets/derived/final_test_layout/"
    "2006_流量_6-16_汉江区/station_tables_daily"
)


def build_alignment_manifest(
    dataset_dir: Path,
    image_dir: Path,
    output_path: Path,
    audit_path: Path,
) -> dict[str, object]:
    return build_real_flow_alignment_manifest(
        dataset_dir=dataset_dir,
        image_dir=image_dir,
        output_path=output_path,
        audit_path=audit_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a scored CSV-to-JPG alignment manifest for real 2006 flow tables."
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/流量/2006"))
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests/flow_real_test_aligned.jsonl"),
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("data/manifests/flow_real_test_alignment_audit.json"),
    )
    args = parser.parse_args()

    audit_payload = build_alignment_manifest(
        dataset_dir=args.dataset_dir,
        image_dir=args.image_dir,
        output_path=args.output,
        audit_path=args.audit_output,
    )
    print(json.dumps(audit_payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
