from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.common.yearbook_flow_common import (
    parse_station_meta,
    read_csv_text,
    sample_id_from_name,
)


def build_manifest(dataset_dir: Path, pdf_path: Path, output_path: Path) -> int:
    records = []
    for csv_path in sorted(dataset_dir.glob("*.csv")):
        csv_text, encoding = read_csv_text(csv_path)
        meta = parse_station_meta(csv_path)
        records.append(
            {
                "sample_id": sample_id_from_name(meta.filename_stem),
                "station_name": meta.station_name,
                "river": meta.river,
                "year": meta.year,
                "pdf_path": str(pdf_path.as_posix()),
                "csv_path": str(csv_path.as_posix()),
                "csv_encoding": encoding,
                "target_csv": csv_text,
                "split": "test",
                "source": "real",
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build real flow test manifest.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/流量/2006"))
    parser.add_argument(
        "--pdf-path",
        type=Path,
        default=Path("datasets/流量/2006 流量 6-16 汉江区（汉江下游水系，汈汊湖、东荆河水系）.pdf"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests/flow_real_test.jsonl"),
    )
    args = parser.parse_args()
    count = build_manifest(args.dataset_dir, args.pdf_path, args.output)
    print(f"Wrote {count} real test records to {args.output}")


if __name__ == "__main__":
    main()

