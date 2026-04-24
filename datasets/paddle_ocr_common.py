from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from yearbook_ocr.ocr.rapidocr_paddle import *  # noqa: F401,F403
