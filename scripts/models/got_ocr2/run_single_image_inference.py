#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from scripts.models.got_ocr2.run_real_inference import (
    DEFAULT_BASE_MODEL,
    DEFAULT_CACHE_ROOT,
    DEFAULT_PROMPT,
    DEFAULT_SINGLE_QUERY,
    GOTImageEvalProcessor,
    infer_one,
    load_model,
    resolve_device,
    resolve_dtype,
    set_cache_env,
)


def run_with_model_chat(model, tokenizer, image: Path, query_mode: str) -> str:
    if query_mode != "official_format":
        raise ValueError("model.chat only supports --query-mode official_format.")
    chat_fn = getattr(model, "chat", None)
    if chat_fn is None and hasattr(model, "base_model"):
        chat_fn = getattr(model.base_model, "chat", None)
    if chat_fn is None:
        raise RuntimeError("Loaded model does not expose chat().")
    return chat_fn(tokenizer, image.as_posix(), "format")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GOT-OCR2.0 on a single image.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--backend", choices=("official_chat", "local_generate"), default="official_chat")
    parser.add_argument(
        "--query-mode",
        choices=("official_format", "custom_default"),
        default="official_format",
    )
    parser.add_argument("--query", type=str)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="auto", choices=("auto", "float16", "bfloat16", "float32"))
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    set_cache_env(args.cache_root)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    query = args.query
    if query is None:
        query = DEFAULT_SINGLE_QUERY if args.query_mode == "official_format" else DEFAULT_PROMPT

    start_time = time.time()
    model, tokenizer = load_model(args.base_model, args.adapter_dir, device, dtype)
    if args.backend == "official_chat":
        prediction = run_with_model_chat(model, tokenizer, args.image, args.query_mode)
    else:
        image_processor = GOTImageEvalProcessor(image_size=1024)
        prediction = infer_one(
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            query=query,
            image_path=args.image,
            max_new_tokens=args.max_new_tokens,
        )
    payload = {
        "image_path": args.image.as_posix(),
        "backend": args.backend,
        "query_mode": args.query_mode,
        "query": query,
        "prediction": prediction,
        "device": device,
        "dtype": str(dtype),
        "elapsed_sec": round(time.time() - start_time, 3),
    }
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
