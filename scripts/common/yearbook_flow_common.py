from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

CSV_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk")
DEFAULT_PROMPT = (
    "Read the table in the image and output only CSV text.\n"
    "Keep the original comma-separated structure.\n"
    "Keep empty rows and empty cells exactly as shown.\n"
    "Do not add explanations, titles, markdown fences, or extra text.\n"
    "Do not normalize number formats.\n"
    "Do not fill in missing values."
)
DEFAULT_GOT_FORMAT_PROMPT = "OCR with format: "


@dataclass(frozen=True)
class StationMeta:
    station_name: str
    year: str
    river: str
    filename_stem: str


def read_csv_text(path: Path) -> tuple[str, str]:
    last_error: UnicodeDecodeError | None = None
    for encoding in CSV_ENCODINGS:
        try:
            return path.read_text(encoding=encoding).replace("\r\n", "\n"), encoding
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Unable to decode {path}")


def parse_csv_text(csv_text: str) -> list[list[str]]:
    return list(csv.reader(csv_text.splitlines()))


def csv_rows_to_text(rows: Iterable[Iterable[str]]) -> str:
    lines = []
    for row in rows:
        lines.append(",".join(str(cell) for cell in row))
    return "\n".join(lines) + "\n"


def csv_rows_to_got_format(rows: Iterable[Iterable[str]]) -> str:
    normalized_rows = [list(row) for row in rows]
    if not normalized_rows:
        return ""
    column_count = max(len(row) for row in normalized_rows)
    spec = "|" + "|".join("c" for _ in range(column_count)) + "|"
    lines = [f"\\begin{{tabular}}{{{spec}}}", "\\hline"]
    for row in normalized_rows:
        padded = row + [""] * (column_count - len(row))
        lines.append(" & ".join(str(cell) for cell in padded) + " \\\\")
        lines.append("\\hline")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def parse_station_meta(csv_path: Path) -> StationMeta:
    stem = csv_path.stem
    if stem.endswith("_已校准"):
        stem = stem[: -len("_已校准")]
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected station filename: {csv_path.name}")
    station_name, year, river = parts[0], parts[1], "_".join(parts[2:])
    return StationMeta(
        station_name=station_name,
        year=year,
        river=river,
        filename_stem=csv_path.stem,
    )


def sample_id_from_name(name: str) -> str:
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:12]
    return f"flow_{digest}"


def is_blank_row(row: list[str]) -> bool:
    return not any(str(cell).strip() for cell in row)


def is_valid_numeric_token(token: str) -> bool:
    token = token.strip()
    if not token:
        return False
    try:
        float(token)
    except ValueError:
        return False
    return True


def month_day_limit(month_index: int) -> int:
    limits = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return limits[month_index]


def seeded_random(seed: int) -> random.Random:
    return random.Random(seed)


def dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
