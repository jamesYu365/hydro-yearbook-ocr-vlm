#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import fitz
from rapid_table_det.inference import TableDetector


DEFAULT_PDFS = [
    "datasets/流量/2006 流量 6-16 汉江区（汉江下游水系，汈汊湖、东荆河水系）.pdf",
    "datasets/水位/2014 水位 6-16 汉江区（汉江下游水系、汈汊湖、东荆河水系）.pdf",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render yearbook PDFs into page images and extract station-level table crops "
            "with rapid_table_det."
        )
    )
    parser.add_argument(
        "--pdf",
        dest="pdf_paths",
        action="append",
        help="PDF path to process. Repeat to pass multiple files. Defaults to the two current target PDFs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("datasets/derived/final_test_layout"),
        help="Root directory for rendered pages, cropped tables, and metadata.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Render DPI for PDF pages.",
    )
    parser.add_argument(
        "--limit-pages",
        type=int,
        default=None,
        help="Only process the first N pages of each PDF. Useful for smoke tests.",
    )
    parser.add_argument(
        "--skip-rendered-pages",
        action="store_true",
        help="Reuse existing rendered page images when present.",
    )
    parser.add_argument(
        "--skip-existing-crops",
        action="store_true",
        help="Skip page-level detection if crops for that page already exist.",
    )
    return parser.parse_args()


def sanitize_stem(name: str) -> str:
    return (
        name.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("（", "(")
        .replace("）", ")")
        .replace("，", ",")
    )


def render_pdf_pages(
    pdf_path: Path,
    page_dir: Path,
    dpi: int,
    limit_pages: int | None,
    skip_rendered_pages: bool,
) -> list[Path]:
    page_dir.mkdir(parents=True, exist_ok=True)
    rendered_paths: list[Path] = []

    with fitz.open(pdf_path) as doc:
        total_pages = doc.page_count
        max_pages = total_pages if limit_pages is None else min(limit_pages, total_pages)
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        for page_index in range(max_pages):
            page_path = page_dir / f"page_{page_index + 1:04d}.jpg"
            rendered_paths.append(page_path)
            if skip_rendered_pages and page_path.exists():
                continue

            page = doc.load_page(page_index)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            pix.save(page_path)

    return rendered_paths


def sort_detections(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        detections,
        key=lambda item: (
            int(item["box"][1]),
            int(item["box"][0]),
        ),
    )


def crop_tables_from_page(
    image_path: Path,
    crop_dir: Path,
    detector: TableDetector,
    skip_existing_crops: bool,
) -> dict[str, Any]:
    crop_dir.mkdir(parents=True, exist_ok=True)
    page_stem = image_path.stem

    if skip_existing_crops and any(crop_dir.glob(f"{page_stem}_table*.jpg")):
        return {
            "page_image_path": image_path.as_posix(),
            "status": "skipped_existing_crops",
            "table_count": None,
            "detections": [],
        }

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to load rendered page image: {image_path}")

    detections, _ = detector(str(image_path), use_cls_det=False)
    detections = sort_detections(detections)

    records: list[dict[str, Any]] = []
    image_height, image_width = image.shape[:2]

    for table_index, detection in enumerate(detections):
        box = [int(value) for value in detection["box"]]
        x1, y1, x2, y2 = box
        x1 = max(0, min(x1, image_width))
        x2 = max(0, min(x2, image_width))
        y1 = max(0, min(y1, image_height))
        y2 = max(0, min(y2, image_height))
        if x2 <= x1 or y2 <= y1:
            continue

        crop = image[y1:y2, x1:x2]
        crop_path = crop_dir / f"{page_stem}_table{table_index}.jpg"
        cv2.imwrite(str(crop_path), crop)

        records.append(
            {
                "table_index": table_index,
                "crop_path": crop_path.as_posix(),
                "box": [x1, y1, x2, y2],
            }
        )

    return {
        "page_image_path": image_path.as_posix(),
        "status": "ok",
        "table_count": len(records),
        "detections": records,
    }


def process_pdf(
    pdf_path: Path,
    output_root: Path,
    detector: TableDetector,
    dpi: int,
    limit_pages: int | None,
    skip_rendered_pages: bool,
    skip_existing_crops: bool,
) -> dict[str, Any]:
    pdf_key = sanitize_stem(pdf_path.stem)
    pdf_output_dir = output_root / pdf_key
    page_dir = pdf_output_dir / "pages"
    crop_dir = pdf_output_dir / "station_tables"
    metadata_path = pdf_output_dir / "layout_detections.json"
    failure_log_path = pdf_output_dir / "layout_failures.jsonl"

    pdf_output_dir.mkdir(parents=True, exist_ok=True)
    rendered_pages = render_pdf_pages(
        pdf_path=pdf_path,
        page_dir=page_dir,
        dpi=dpi,
        limit_pages=limit_pages,
        skip_rendered_pages=skip_rendered_pages,
    )

    page_records: list[dict[str, Any]] = []
    failure_records: list[dict[str, Any]] = []

    for page_number, page_path in enumerate(rendered_pages, start=1):
        try:
            page_record = crop_tables_from_page(
                image_path=page_path,
                crop_dir=crop_dir,
                detector=detector,
                skip_existing_crops=skip_existing_crops,
            )
            page_record["page_number"] = page_number
            if page_record["status"] == "ok" and page_record["table_count"] == 0:
                failure_records.append(
                    {
                        "pdf_path": pdf_path.as_posix(),
                        "page_number": page_number,
                        "page_image_path": page_path.as_posix(),
                        "reason": "no_table_detected",
                    }
                )
            page_records.append(page_record)
        except Exception as exc:
            failure_records.append(
                {
                    "pdf_path": pdf_path.as_posix(),
                    "page_number": page_number,
                    "page_image_path": page_path.as_posix(),
                    "reason": str(exc),
                }
            )

    metadata = {
        "pdf_path": pdf_path.as_posix(),
        "dpi": dpi,
        "page_count": len(rendered_pages),
        "pages": page_records,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    with failure_log_path.open("w", encoding="utf-8") as handle:
        for record in failure_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    total_tables = sum(
        page_record["table_count"]
        for page_record in page_records
        if page_record["status"] == "ok" and page_record["table_count"] is not None
    )
    return {
        "pdf_path": pdf_path.as_posix(),
        "output_dir": pdf_output_dir.as_posix(),
        "page_count": len(rendered_pages),
        "table_count": total_tables,
        "failure_count": len(failure_records),
    }


def main() -> None:
    args = parse_args()
    pdf_paths = [Path(path) for path in (args.pdf_paths or DEFAULT_PDFS)]

    for pdf_path in pdf_paths:
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

    detector = TableDetector()
    summaries = []
    for pdf_path in pdf_paths:
        summary = process_pdf(
            pdf_path=pdf_path,
            output_root=args.output_root,
            detector=detector,
            dpi=args.dpi,
            limit_pages=args.limit_pages,
            skip_rendered_pages=args.skip_rendered_pages,
            skip_existing_crops=args.skip_existing_crops,
        )
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False))

    summary_path = args.output_root / "run_summary.json"
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
