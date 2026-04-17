#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.real_flow_test_prep import (
    parse_csv_entry,
    parse_image_entry,
    resolve_reciprocal_matches,
)
from scripts.common.yearbook_flow_common import read_csv_text


DEFAULT_IMAGE_DIR = Path(
    "datasets/derived/final_test_layout/"
    "2006_流量_6-16_汉江区(汉江下游水系,汈汊湖、东荆河水系)/station_tables_daily"
)


def build_alignment_manifest(
    dataset_dir: Path,
    image_dir: Path,
    output_path: Path,
    audit_path: Path,
) -> dict[str, object]:
    csv_entries = [parse_csv_entry(path) for path in sorted(dataset_dir.glob("*.csv"))]
    image_entries = [parse_image_entry(path) for path in sorted(image_dir.glob("*.jpg"))]
    rows, audit = resolve_reciprocal_matches(csv_entries, image_entries)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            csv_text, csv_encoding = read_csv_text(Path(row["csv_path"]))
            payload = {
                "sample_id": row["sample_id"],
                "station_name": row["station_name_csv"],
                "river": row["river_csv"],
                "year": row["year"],
                "csv_path": row["csv_path"],
                "csv_encoding": csv_encoding,
                "image_path": row["image_path"],
                "title_text": row["image_title_text"],
                "target_csv": csv_text,
                "match_score": row["match_score"],
                "match_status": row["match_status"],
                "split": "test",
                "source": "real",
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_payload = {
        "dataset_dir": dataset_dir.as_posix(),
        "image_dir": image_dir.as_posix(),
        **audit,
        "rows": rows,
    }
    audit_path.write_text(
        json.dumps(audit_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return audit_payload


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
