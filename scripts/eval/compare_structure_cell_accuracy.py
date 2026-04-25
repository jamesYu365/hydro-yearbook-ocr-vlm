#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from yearbook_ocr.eval.model_comparison import main


if __name__ == "__main__":
    main()
