#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
from rapidocr_onnxruntime import RapidOCR

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.real_flow_test_prep import ocr_tokens_from_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RapidOCR on a single image and print token-level OCR results."
    )
    parser.add_argument("image", type=Path, help="Image path to recognize.")
    parser.add_argument(
        "--ocr-det-model-path",
        type=Path,
        default=Path("./datasets/ONNX_models/ch_PP-OCRv5_det_server.onnx"),
        help="Optional RapidOCR detection ONNX model path.",
    )
    parser.add_argument(
        "--ocr-rec-model-path",
        type=Path,
        default=Path("./datasets/ONNX_models/ch_PP-OCRv5_rec_server.onnx"),
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
        choices=("auto", "on", "off"),
        default="auto",
        help="RapidOCR device policy. 'auto' tries CUDA first and falls back to CPU.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional JSON output path for the OCR payload.",
    )
    return parser.parse_args()


def init_ocr_engine(args: argparse.Namespace) -> RapidOCR:
    kwargs: dict[str, Any] = {
        "max_side_len": args.ocr_max_side_len,
        "det_model_path": str(args.ocr_det_model_path),
        "rec_model_path": str(args.ocr_rec_model_path),
    }

    if args.ocr_cuda == "off":
        return RapidOCR(det_use_cuda=False, rec_use_cuda=False, **kwargs)
    if args.ocr_cuda == "on":
        return RapidOCR(det_use_cuda=True, rec_use_cuda=True, **kwargs)

    try:
        return RapidOCR(det_use_cuda=True, rec_use_cuda=True, **kwargs)
    except Exception:
        return RapidOCR(det_use_cuda=False, rec_use_cuda=False, **kwargs)


def main() -> None:
    args = parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        raise RuntimeError(f"Failed to load image: {args.image}")

    ocr_engine = init_ocr_engine(args)
    ocr_result, _ = ocr_engine(image)
    tokens = ocr_tokens_from_result(ocr_result)

    payload = {
        "image_path": args.image.as_posix(),
        "image_height": int(image.shape[0]),
        "image_width": int(image.shape[1]),
        "token_count": len(tokens),
        "tokens": [
            {
                "text": token.text,
                "top": token.top,
                "bottom": token.bottom,
                "left": token.left,
                "right": token.right,
            }
            for token in tokens
        ],
    }

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
