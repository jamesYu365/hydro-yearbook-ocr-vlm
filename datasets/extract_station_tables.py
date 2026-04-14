#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import cv2
import fitz
from rapid_table_det.inference import TableDetector
from rapidocr_onnxruntime import RapidOCR


DEFAULT_PDFS = [
    "datasets/流量/2006 流量 6-16 汉江区（汉江下游水系，汈汊湖、东荆河水系）.pdf",
    "datasets/水位/2014 水位 6-16 汉江区（汉江下游水系、汈汊湖、东荆河水系）.pdf",
]

FLOW_KEYWORD = "逐日平均流量表"
WATER_LEVEL_KEYWORD = "逐日平均水位表"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render yearbook PDFs into page images, extract plain and buffered table crops, "
            "and run title OCR on a title ROI built from the top buffer strip plus a small "
            "downward compensation."
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
        help="Root directory for rendered pages, cropped tables, title ROI images, and metadata.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Render DPI for PDF pages.")
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
        help="Skip page-level extraction if final-named outputs already exist for that page.",
    )
    parser.add_argument(
        "--title-buffer-px",
        type=int,
        default=180,
        help="Fixed number of pixels added above the detected table box for title recovery.",
    )
    parser.add_argument(
        "--title-bottom-buffer-px",
        type=int,
        default=12,
        help="Fixed number of pixels added below the original table top when extracting the title ROI.",
    )
    parser.add_argument(
        "--ocr-det-model-path",
        type=Path,
        default=None,
        help="Optional RapidOCR detection ONNX model path.",
    )
    parser.add_argument(
        "--ocr-rec-model-path",
        type=Path,
        default=None,
        help="Optional RapidOCR recognition ONNX model path.",
    )
    parser.add_argument(
        "--ocr-max-side-len",
        type=int,
        default=10000,
        help="RapidOCR max side length.",
    )
    parser.add_argument(
        "--ocr-cuda",
        choices=["auto", "on", "off"],
        default="auto",
        help="RapidOCR device policy. 'auto' tries CUDA first and falls back to CPU.",
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


def sanitize_title_for_filename(title: str, fallback: str, max_length: int = 100) -> str:
    sanitized = title.strip()
    sanitized = sanitized.replace("（", "(").replace("）", ")")
    sanitized = re.sub(r"\s+", "", sanitized)
    sanitized = re.sub(r'[\\/:*?"<>|]', "_", sanitized)
    sanitized = re.sub(r"_+", "_", sanitized).strip("._")
    if not sanitized:
        sanitized = fallback
    return sanitized[:max_length] or fallback


def extract_year_from_pdf_key(pdf_key: str) -> str:
    match = re.match(r"(\d{4})_", pdf_key)
    if match is None:
        raise ValueError(f"Unable to extract year from output folder key: {pdf_key}")
    return match.group(1)


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


def init_ocr_engine(args: argparse.Namespace) -> RapidOCR:
    kwargs: dict[str, Any] = {"max_side_len": args.ocr_max_side_len}
    if args.ocr_det_model_path is not None:
        kwargs["det_model_path"] = str(args.ocr_det_model_path)
    if args.ocr_rec_model_path is not None:
        kwargs["rec_model_path"] = str(args.ocr_rec_model_path)

    if args.ocr_cuda == "off":
        return RapidOCR(det_use_cuda=False, rec_use_cuda=False, **kwargs)
    if args.ocr_cuda == "on":
        return RapidOCR(det_use_cuda=True, rec_use_cuda=True, **kwargs)

    try:
        return RapidOCR(det_use_cuda=True, rec_use_cuda=True, **kwargs)
    except Exception:
        return RapidOCR(det_use_cuda=False, rec_use_cuda=False, **kwargs)


def expected_title_keyword(pdf_path: Path) -> str:
    if "流量" in pdf_path.as_posix():
        return FLOW_KEYWORD
    if "水位" in pdf_path.as_posix():
        return WATER_LEVEL_KEYWORD
    raise ValueError(f"Unable to infer expected title keyword from PDF path: {pdf_path}")


def merge_title_lines(ocr_result: list[Any]) -> tuple[list[str], str]:
    line_items: list[tuple[float, float, str]] = []
    for entry in ocr_result:
        if len(entry) < 2:
            continue
        points = entry[0]
        text = str(entry[1]).strip()
        if not text:
            continue
        y_min = min(point[1] for point in points)
        x_min = min(point[0] for point in points)
        line_items.append((y_min, x_min, text))

    ordered_texts = [item[2] for item in sorted(line_items, key=lambda item: (item[0], item[1]))]
    merged_text = "".join(ordered_texts)
    return ordered_texts, merged_text


def recognize_table_title(
    title_region: Any,
    ocr_engine: RapidOCR,
    expected_keyword: str,
) -> dict[str, Any]:
    ocr_result, _ = ocr_engine(title_region)
    ordered_lines, merged_text = merge_title_lines(ocr_result or [])
    success = expected_keyword in merged_text
    return {
        "ocr_text_lines": ordered_lines,
        "ocr_merged_text": merged_text,
        "recognized_title_raw": merged_text if success else None,
        "title_ocr_success": success,
        "title_failure_reason": None if success else "expected_keyword_missing",
    }


def extract_title_prefix(recognized_title: str, expected_keyword: str) -> str:
    match = re.search(rf"^(.*?){re.escape(expected_keyword)}", recognized_title)
    if match is None:
        return ""
    prefix = match.group(1)
    prefix = re.sub(r"^\d+", "", prefix)
    prefix = prefix.strip(" _-:：,，.。")
    return prefix.strip()


def ensure_clean_output_dirs(*dirs: Path) -> None:
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)


def remove_existing_page_outputs(page_stem: str, *dirs: Path) -> None:
    for directory in dirs:
        for path in directory.glob(f"{page_stem}_table*.jpg"):
            path.unlink()


def remove_obsolete_dirs(pdf_output_dir: Path) -> None:
    obsolete_dirs = (
        pdf_output_dir / "station_tables",
        pdf_output_dir / "title_buffer_regions",
        pdf_output_dir / "station_tables_plain_titled",
        pdf_output_dir / "station_tables_buffered_titled",
    )
    for directory in obsolete_dirs:
        if not directory.exists() or not directory.is_dir():
            continue
        for child in directory.iterdir():
            if child.is_file():
                child.unlink()
        directory.rmdir()


def crop_tables_from_page(
    image_path: Path,
    plain_crop_dir: Path,
    buffered_crop_dir: Path,
    title_roi_dir: Path,
    detector: TableDetector,
    ocr_engine: RapidOCR,
    pdf_path: Path,
    year: str,
    title_buffer_px: int,
    title_bottom_buffer_px: int,
    skip_existing_crops: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ensure_clean_output_dirs(plain_crop_dir, buffered_crop_dir, title_roi_dir)

    page_stem = image_path.stem
    existing_plain = list(plain_crop_dir.glob(f"{page_stem}_table*.jpg"))
    existing_buffered = list(buffered_crop_dir.glob(f"{page_stem}_table*.jpg"))
    existing_title_rois = list(title_roi_dir.glob(f"{page_stem}_table*.jpg"))
    if skip_existing_crops and existing_plain and existing_buffered and existing_title_rois:
        return (
            {
                "page_image_path": image_path.as_posix(),
                "status": "skipped_existing_crops",
                "table_count": None,
                "title_ocr_success_count": None,
                "detections": [],
            },
            [],
        )

    remove_existing_page_outputs(page_stem, plain_crop_dir, buffered_crop_dir, title_roi_dir)

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to load rendered page image: {image_path}")

    detections, _ = detector(str(image_path), use_cls_det=False)
    detections = sort_detections(detections)
    expected_keyword = expected_title_keyword(pdf_path)
    image_height, image_width = image.shape[:2]

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for table_index, detection in enumerate(detections):
        x1, y1, x2, y2 = [int(value) for value in detection["box"]]
        x1 = max(0, min(x1, image_width))
        x2 = max(0, min(x2, image_width))
        y1 = max(0, min(y1, image_height))
        y2 = max(0, min(y2, image_height))
        if x2 <= x1 or y2 <= y1:
            failures.append(
                {
                    "pdf_path": pdf_path.as_posix(),
                    "page_image_path": image_path.as_posix(),
                    "table_index": table_index,
                    "reason": "invalid_original_box",
                }
            )
            continue

        buffered_y1 = max(0, y1 - title_buffer_px)
        title_roi_y2 = min(image_height, y1 + title_bottom_buffer_px)
        stable_name = f"{page_stem}_table{table_index}"
        final_basename = f"{stable_name}_{year}"

        title_metadata = {
            "ocr_text_lines": [],
            "ocr_merged_text": "",
            "recognized_title_raw": None,
            "recognized_title_clean": None,
            "final_basename": final_basename,
            "title_ocr_success": False,
            "title_failure_reason": None,
        }

        if title_roi_y2 <= buffered_y1:
            title_metadata["title_failure_reason"] = "empty_title_roi"
            failures.append(
                {
                    "pdf_path": pdf_path.as_posix(),
                    "page_image_path": image_path.as_posix(),
                    "table_index": table_index,
                    "reason": "empty_title_roi",
                }
            )
        else:
            title_roi = image[buffered_y1:title_roi_y2, x1:x2]
            try:
                ocr_metadata = recognize_table_title(
                    title_region=title_roi,
                    ocr_engine=ocr_engine,
                    expected_keyword=expected_keyword,
                )
                title_metadata.update(ocr_metadata)
                if ocr_metadata["title_ocr_success"]:
                    cleaned_title = sanitize_title_for_filename(
                        title=extract_title_prefix(
                            recognized_title=ocr_metadata["recognized_title_raw"] or "",
                            expected_keyword=expected_keyword,
                        ),
                        fallback="unknown_title",
                    )
                    title_metadata["recognized_title_clean"] = cleaned_title
                    title_metadata["final_basename"] = sanitize_title_for_filename(
                        title=f"{stable_name}_{cleaned_title}_{year}",
                        fallback=final_basename,
                    )
                else:
                    failures.append(
                        {
                            "pdf_path": pdf_path.as_posix(),
                            "page_image_path": image_path.as_posix(),
                            "table_index": table_index,
                            "reason": ocr_metadata["title_failure_reason"],
                            "expected_keyword": expected_keyword,
                            "ocr_merged_text": ocr_metadata["ocr_merged_text"],
                        }
                    )
            except Exception as exc:
                title_metadata["title_failure_reason"] = "title_ocr_runtime_error"
                failures.append(
                    {
                        "pdf_path": pdf_path.as_posix(),
                        "page_image_path": image_path.as_posix(),
                        "table_index": table_index,
                        "reason": "title_ocr_runtime_error",
                        "details": str(exc),
                    }
                )

        final_basename = title_metadata["final_basename"]
        plain_crop = image[y1:y2, x1:x2]
        buffered_crop = image[buffered_y1:y2, x1:x2]
        title_roi = image[buffered_y1:title_roi_y2, x1:x2] if title_roi_y2 > buffered_y1 else None

        plain_crop_path = plain_crop_dir / f"{final_basename}.jpg"
        buffered_crop_path = buffered_crop_dir / f"{final_basename}.jpg"
        title_roi_path = title_roi_dir / f"{final_basename}.jpg"
        cv2.imwrite(str(plain_crop_path), plain_crop)
        cv2.imwrite(str(buffered_crop_path), buffered_crop)
        if title_roi is not None:
            cv2.imwrite(str(title_roi_path), title_roi)

        records.append(
            {
                "table_index": table_index,
                "stable_name": stable_name,
                "year": year,
                "original_box": [x1, y1, x2, y2],
                "buffered_box": [x1, buffered_y1, x2, y2],
                "title_roi_box": [x1, buffered_y1, x2, title_roi_y2],
                "title_top_buffer_px": title_buffer_px,
                "title_bottom_buffer_px": title_bottom_buffer_px,
                "title_keyword_expected": expected_keyword,
                "plain_crop_path": plain_crop_path.as_posix(),
                "buffered_crop_path": buffered_crop_path.as_posix(),
                "title_roi_path": title_roi_path.as_posix() if title_roi is not None else None,
                "ocr_text_lines": title_metadata["ocr_text_lines"],
                "ocr_merged_text": title_metadata["ocr_merged_text"],
                "recognized_title_raw": title_metadata["recognized_title_raw"],
                "recognized_title_clean": title_metadata["recognized_title_clean"],
                "final_basename": title_metadata["final_basename"],
                "title_ocr_success": title_metadata["title_ocr_success"],
                "title_failure_reason": title_metadata["title_failure_reason"],
            }
        )

    return (
        {
            "page_image_path": image_path.as_posix(),
            "status": "ok",
            "table_count": len(records),
            "title_ocr_success_count": sum(1 for record in records if record["title_ocr_success"]),
            "detections": records,
        },
        failures,
    )


def process_pdf(
    pdf_path: Path,
    output_root: Path,
    detector: TableDetector,
    ocr_engine: RapidOCR,
    dpi: int,
    limit_pages: int | None,
    skip_rendered_pages: bool,
    skip_existing_crops: bool,
    title_buffer_px: int,
    title_bottom_buffer_px: int,
) -> dict[str, Any]:
    pdf_key = sanitize_stem(pdf_path.stem)
    year = extract_year_from_pdf_key(pdf_key)
    pdf_output_dir = output_root / pdf_key
    page_dir = pdf_output_dir / "pages"
    plain_crop_dir = pdf_output_dir / "station_tables_plain"
    buffered_crop_dir = pdf_output_dir / "station_tables_buffered"
    title_roi_dir = pdf_output_dir / "title_rois"
    metadata_path = pdf_output_dir / "layout_detections.json"
    failure_log_path = pdf_output_dir / "layout_failures.jsonl"

    pdf_output_dir.mkdir(parents=True, exist_ok=True)
    remove_obsolete_dirs(pdf_output_dir)
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
            page_record, page_failures = crop_tables_from_page(
                image_path=page_path,
                plain_crop_dir=plain_crop_dir,
                buffered_crop_dir=buffered_crop_dir,
                title_roi_dir=title_roi_dir,
                detector=detector,
                ocr_engine=ocr_engine,
                pdf_path=pdf_path,
                year=year,
                title_buffer_px=title_buffer_px,
                title_bottom_buffer_px=title_bottom_buffer_px,
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
            for failure in page_failures:
                failure["page_number"] = page_number
                failure_records.append(failure)
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

    total_tables = sum(
        page_record["table_count"]
        for page_record in page_records
        if page_record["status"] == "ok" and page_record["table_count"] is not None
    )
    total_title_ocr_success = sum(
        page_record["title_ocr_success_count"]
        for page_record in page_records
        if page_record["status"] == "ok" and page_record["title_ocr_success_count"] is not None
    )
    title_ocr_failure_count = sum(
        1
        for failure in failure_records
        if failure["reason"] in {"expected_keyword_missing", "empty_title_roi", "title_ocr_runtime_error"}
    )

    metadata = {
        "pdf_path": pdf_path.as_posix(),
        "dpi": dpi,
        "year": year,
        "title_buffer_px": title_buffer_px,
        "title_bottom_buffer_px": title_bottom_buffer_px,
        "title_keyword_expected": expected_title_keyword(pdf_path),
        "page_count": len(rendered_pages),
        "pages": page_records,
        "summary": {
            "table_count": total_tables,
            "title_ocr_success_count": total_title_ocr_success,
            "title_ocr_failure_count": title_ocr_failure_count,
            "failure_count": len(failure_records),
        },
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    with failure_log_path.open("w", encoding="utf-8") as handle:
        for record in failure_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "pdf_path": pdf_path.as_posix(),
        "output_dir": pdf_output_dir.as_posix(),
        "page_count": len(rendered_pages),
        "table_count": total_tables,
        "title_ocr_success_count": total_title_ocr_success,
        "title_ocr_failure_count": title_ocr_failure_count,
        "failure_count": len(failure_records),
    }


def main() -> None:
    args = parse_args()
    pdf_paths = [Path(path) for path in (args.pdf_paths or DEFAULT_PDFS)]

    for pdf_path in pdf_paths:
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

    detector = TableDetector()
    ocr_engine = init_ocr_engine(args)
    summaries = []
    for pdf_path in pdf_paths:
        summary = process_pdf(
            pdf_path=pdf_path,
            output_root=args.output_root,
            detector=detector,
            ocr_engine=ocr_engine,
            dpi=args.dpi,
            limit_pages=args.limit_pages,
            skip_rendered_pages=args.skip_rendered_pages,
            skip_existing_crops=args.skip_existing_crops,
            title_buffer_px=args.title_buffer_px,
            title_bottom_buffer_px=args.title_bottom_buffer_px,
        )
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False))

    summary_path = args.output_root / "run_summary.json"
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
