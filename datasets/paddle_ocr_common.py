from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _build_params(*, max_side_len: int, use_cuda: bool) -> dict[str, Any]:
    from rapidocr import EngineType, LangDet, LangRec, ModelType, OCRVersion

    return {
        "Global.max_side_len": max_side_len,
        "Det.engine_type": EngineType.PADDLE,
        "Det.lang_type": LangDet.CH,
        "Det.model_type": ModelType.SERVER,
        "Det.ocr_version": OCRVersion.PPOCRV5,
        "Cls.engine_type": EngineType.PADDLE,
        "Cls.model_type": ModelType.SERVER,
        "Cls.ocr_version": OCRVersion.PPOCRV5,
        "Rec.engine_type": EngineType.PADDLE,
        "Rec.lang_type": LangRec.CH,
        "Rec.model_type": ModelType.SERVER,
        "Rec.ocr_version": OCRVersion.PPOCRV5,
        "EngineConfig.paddle.use_cuda": use_cuda,
        "EngineConfig.paddle.cuda_ep_cfg.device_id": 0,
    }


def init_paddle_ocr_engine(
    *,
    max_side_len: int,
    ocr_cuda: str = "on",
    ocr_device_id: int = 1,
) -> Any:
    if ocr_cuda != "off":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(ocr_device_id)

    import paddle
    from rapidocr import RapidOCR

    if ocr_cuda == "off":
        return RapidOCR(params=_build_params(max_side_len=max_side_len, use_cuda=False))

    if ocr_cuda == "on":
        paddle.set_device("gpu:0")
        return RapidOCR(params=_build_params(max_side_len=max_side_len, use_cuda=True))

    try:
        paddle.set_device("gpu:0")
        return RapidOCR(params=_build_params(max_side_len=max_side_len, use_cuda=True))
    except Exception:
        return RapidOCR(params=_build_params(max_side_len=max_side_len, use_cuda=False))


def run_paddle_ocr(ocr_engine: Any, image: str | Path | Any) -> list[list[Any]]:
    result = ocr_engine(image)
    boxes = [] if result.boxes is None else result.boxes
    txts = () if result.txts is None else result.txts
    scores = () if result.scores is None else result.scores

    normalized: list[list[Any]] = []
    for box, text, score in zip(boxes, txts, scores):
        normalized.append([box.tolist(), str(text), float(score)])
    return normalized
