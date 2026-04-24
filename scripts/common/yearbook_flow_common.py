from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from yearbook_ocr.common.jsonl import write_json as dump_json
from yearbook_ocr.common.tabular import *  # noqa: F401,F403
