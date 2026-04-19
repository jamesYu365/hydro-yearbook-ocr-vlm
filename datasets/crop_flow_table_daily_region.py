#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.real_flow_test_prep import (
    apply_bottom_buffer,
    fallback_cut_y_from_lines,
    find_horizontal_cut_y,
    ocr_tokens_from_result,
    select_statistics_anchor,
)
from datasets.paddle_ocr_common import init_paddle_ocr_engine, run_paddle_ocr


DEFAULT_PLAIN_DIR = Path(
    "datasets/derived/final_test_layout/"
    "2006_流量_6-16_汉江区/station_tables_plain"
)
DEFAULT_OUTPUT_DIR = DEFAULT_PLAIN_DIR.parent / "station_tables_daily"
DEFAULT_METADATA_PATH = DEFAULT_PLAIN_DIR.parent / "station_tables_daily_metadata.json"


def init_ocr_engine(
    max_side_len: int,
    ocr_cuda: str,
    ocr_device_id: int,
) -> Any:
    return init_paddle_ocr_engine(
        max_side_len=max_side_len,
        ocr_cuda=ocr_cuda,
        ocr_device_id=ocr_device_id,
    )


def multi_threshold_line_cut(row_dark_counts: list[int], width: int, height: int) -> tuple[int | None, str | None]:
    attempts = (
        (0.60, 0.60),
        (0.55, 0.60),
        (0.50, 0.55),
        (0.40, 0.55),
        (0.35, 0.55),
    )
    for min_ratio, start_ratio in attempts:
        cut_y = fallback_cut_y_from_lines(
            row_dark_counts,
            width,
            height,
            min_ratio=min_ratio,
            start_ratio=start_ratio,
        )
        if cut_y is not None:
            return cut_y, f"line_fallback_{min_ratio:.2f}_{start_ratio:.2f}"
    return None, None


def crop_daily_region(
    image_path: Path,
    output_dir: Path,
    ocr_engine: Any | None,
    bottom_buffer_px: int,
    debug_roi_dir: Path | None = None,
) -> dict[str, Any]:
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to load image: {image_path}")
    height, width = image.shape[:2]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / image_path.name
    if output_path.exists():
        output_path.unlink()

    tokens = []
    anchor = None
    anchor_method = None
    if ocr_engine is not None:
        roi_top = int(height * 0.68)
        roi_right = max(int(width * 0.22), 180)
        lower_label_region = image[roi_top:, :roi_right]
        if debug_roi_dir is not None:
            debug_roi_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_roi_dir / image_path.name), lower_label_region)
        ocr_result = run_paddle_ocr(ocr_engine, lower_label_region)
        tokens = ocr_tokens_from_result(ocr_result, y_offset=roi_top)
        anchor, anchor_method = select_statistics_anchor(tokens)

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    row_dark_counts = (binary.sum(axis=1) / 255).astype(int).tolist()

    cut_y = None
    cut_method = None
    if anchor is not None:
        cut_y = find_horizontal_cut_y(row_dark_counts, width, anchor.top)
        cut_method = "anchor_line" if cut_y is not None else None
        if cut_y is None:
            cut_y = max(int(anchor.top) - 6, 1)
            cut_method = f"{anchor_method}_direct"

    if cut_y is None and ocr_engine is None:
        cut_y, cut_method = multi_threshold_line_cut(row_dark_counts, width, height)

    if cut_y is None:
        raise RuntimeError(f"Unable to determine cut boundary for {image_path.name}: no_valid_average_anchor")

    cut_y = apply_bottom_buffer(cut_y, height, bottom_buffer_px=bottom_buffer_px)

    cropped = image[:cut_y, :]
    cv2.imwrite(str(output_path), cropped)

    return {
        "source_image_path": image_path.as_posix(),
        "output_image_path": output_path.as_posix(),
        "image_height": height,
        "image_width": width,
        "cut_y": cut_y,
        "cut_method": cut_method,
        "anchor_method": anchor_method,
        "anchor_text": anchor.text if anchor is not None else None,
        "anchor_top": anchor.top if anchor is not None else None,
        "detected_tokens": [
            {
                "text": token.text,
                "top": token.top,
                "bottom": token.bottom,
                "left": token.left,
                "right": token.right,
            }
            for token in tokens
            if "平均" in token.text or "年统计" in token.text
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crop plain real flow table images down to the daily-only region."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_PLAIN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--disable-ocr",
        action="store_true",
        help="Experimental fallback path. Do not use for the main real-data pipeline unless explicitly approved.",
    )
    parser.add_argument(
        "--ocr-max-side-len", type=int, default=10000
    )
    parser.add_argument(
        "--ocr-cuda",
        choices=("on", "off", "auto"),
        default="on",
        help="RapidOCR Paddle device policy. Default uses GPU 1 via CUDA_VISIBLE_DEVICES remapping.",
    )
    parser.add_argument(
        "--ocr-device-id",
        type=int,
        default=1,
        help="Physical GPU id to expose to Paddle. The script remaps this card to gpu:0 inside the process.",
    )
    parser.add_argument(
        "--bottom-buffer-px",
        type=int,
        default=6,
        help="Extend the crop a few pixels below the detected divider to avoid clipping the 31st row.",
    )
    parser.add_argument(
        "--debug-roi-dir",
        type=Path,
        default=None,
        help="Optional directory for saving the OCR ROI images used to detect statistics anchors.",
    )
    args = parser.parse_args()

    ocr_engine = None
    if not args.disable_ocr:
        ocr_engine = init_ocr_engine(
            max_side_len=args.ocr_max_side_len,
            ocr_cuda=args.ocr_cuda,
            ocr_device_id=args.ocr_device_id,
        )

    records = []
    failures = []
    image_paths = sorted(args.input_dir.glob("*.jpg"))
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    for image_path in image_paths:
        try:
            records.append(
                crop_daily_region(
                    image_path=image_path,
                    output_dir=args.output_dir,
                    ocr_engine=ocr_engine,
                    bottom_buffer_px=args.bottom_buffer_px,
                    debug_roi_dir=args.debug_roi_dir,
                )
            )
        except Exception as exc:
            failures.append(
                {
                    "source_image_path": image_path.as_posix(),
                    "reason": str(exc),
                }
            )

    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "input_dir": args.input_dir.as_posix(),
        "output_dir": args.output_dir.as_posix(),
        "summary": {
            "input_count": len(image_paths),
            "cropped_count": len(records),
            "failure_count": len(failures),
        },
        "records": records,
        "failures": failures,
    }
    args.metadata_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
