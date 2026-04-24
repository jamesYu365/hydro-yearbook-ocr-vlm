from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from yearbook_ocr.common.tabular import parse_station_meta, sample_id_from_name


PHRASE_REPLACEMENTS = (
    ("（", "("),
    ("）", ")"),
    (" ", ""),
    ("\u3000", ""),
    ("_", ""),
    ("－", "-"),
    ("環", "澴"),
    ("不花园", "花园"),
    ("月万福闸", "万福闸"),
    ("涢水安陆", "涢水安陆"),
    ("水安陆", "涢水安陆"),
    ("水澴潭", "涢水澴潭"),
    ("水環潭", "涢水澴潭"),
    ("温水随州", "涢水随州"),
    ("溫水随州", "涢水随州"),
)

PURE_ORDINAL_PATTERN = re.compile(r"\(([一二三四五六七八九十]+)\)")
IMAGE_STEM_PATTERN = re.compile(r"^(page_\d+_table\d+)_(.+)_(\d{4})$")


@dataclass(frozen=True)
class CsvEntry:
    sample_id: str
    csv_path: Path
    station_name: str
    river: str
    year: str


@dataclass(frozen=True)
class ImageEntry:
    image_path: Path
    stable_name: str
    title_text: str
    year: str


@dataclass(frozen=True)
class MatchCandidate:
    csv_entry: CsvEntry
    image_entry: ImageEntry
    score: float
    station_similarity: float
    river_similarity: float
    combined_similarity: float


@dataclass(frozen=True)
class OcrToken:
    text: str
    left: float
    top: float
    right: float
    bottom: float

    @property
    def center_y(self) -> float:
        return (self.top + self.bottom) / 2.0


def normalize_text(text: str) -> str:
    normalized = text.strip()
    for source, target in PHRASE_REPLACEMENTS:
        normalized = normalized.replace(source, target)
    if normalized.endswith("站"):
        normalized = normalized[:-1]
    normalized = PURE_ORDINAL_PATTERN.sub("", normalized)
    normalized = re.sub(r"[,:：，。.\-]", "", normalized)
    return normalized


def station_key(station_name: str) -> str:
    return normalize_text(station_name)


def river_key(river: str) -> str:
    return normalize_text(river)


def image_title_key(title_text: str) -> str:
    normalized = normalize_text(title_text)
    return re.sub(r"^page\d+table\d+", "", normalized)


def similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    if left in right or right in left:
        shorter = min(len(left), len(right))
        longer = max(len(left), len(right))
        return shorter / max(longer, 1)
    return SequenceMatcher(None, left, right).ratio()


def parse_csv_entry(csv_path: Path) -> CsvEntry:
    meta = parse_station_meta(csv_path)
    return CsvEntry(
        sample_id=sample_id_from_name(meta.filename_stem),
        csv_path=csv_path,
        station_name=meta.station_name,
        river=meta.river,
        year=meta.year,
    )


def parse_image_entry(image_path: Path) -> ImageEntry:
    match = IMAGE_STEM_PATTERN.match(image_path.stem)
    if match is None:
        raise ValueError(f"Unexpected image filename: {image_path.name}")
    stable_name, title_text, year = match.groups()
    return ImageEntry(
        image_path=image_path,
        stable_name=stable_name,
        title_text=title_text,
        year=year,
    )


def score_match(csv_entry: CsvEntry, image_entry: ImageEntry) -> MatchCandidate:
    if csv_entry.year != image_entry.year:
        return MatchCandidate(csv_entry, image_entry, 0.0, 0.0, 0.0, 0.0)

    station_norm = station_key(csv_entry.station_name)
    river_norm = river_key(csv_entry.river)
    title_norm = image_title_key(image_entry.title_text)
    combined_norm = f"{river_norm}{station_norm}"

    station_similarity = similarity(station_norm, title_norm)
    river_similarity = similarity(river_norm, title_norm)
    combined_similarity = similarity(combined_norm, title_norm)

    score = 0.0
    score += combined_similarity * 70.0
    score += station_similarity * 45.0
    score += river_similarity * 25.0
    if combined_norm == title_norm:
        score += 30.0
    elif combined_norm in title_norm:
        score += 20.0
    if station_norm and station_norm in title_norm:
        score += 20.0
    if river_norm and river_norm in title_norm:
        score += 10.0
    if title_norm.endswith(station_norm):
        score += 5.0

    return MatchCandidate(
        csv_entry=csv_entry,
        image_entry=image_entry,
        score=round(score, 4),
        station_similarity=round(station_similarity, 4),
        river_similarity=round(river_similarity, 4),
        combined_similarity=round(combined_similarity, 4),
    )


def build_candidate_table(
    csv_entries: list[CsvEntry],
    image_entries: list[ImageEntry],
) -> tuple[dict[Path, list[MatchCandidate]], dict[Path, list[MatchCandidate]]]:
    csv_to_candidates: dict[Path, list[MatchCandidate]] = {}
    image_to_candidates: dict[Path, list[MatchCandidate]] = {entry.image_path: [] for entry in image_entries}

    for csv_entry in csv_entries:
        candidates = sorted(
            (score_match(csv_entry, image_entry) for image_entry in image_entries),
            key=lambda item: item.score,
            reverse=True,
        )
        csv_to_candidates[csv_entry.csv_path] = candidates
        for candidate in candidates:
            image_to_candidates[candidate.image_entry.image_path].append(candidate)

    for image_path, candidates in image_to_candidates.items():
        image_to_candidates[image_path] = sorted(candidates, key=lambda item: item.score, reverse=True)

    return csv_to_candidates, image_to_candidates


def resolve_reciprocal_matches(
    csv_entries: list[CsvEntry],
    image_entries: list[ImageEntry],
    min_score: float = 95.0,
    min_csv_margin: float = 8.0,
    min_image_margin: float = 8.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not image_entries:
        raise ValueError("No JPG files were found for alignment.")

    csv_to_candidates, image_to_candidates = build_candidate_table(csv_entries, image_entries)
    rows: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    matched_image_paths: set[Path] = set()

    for csv_entry in csv_entries:
        ranked = csv_to_candidates[csv_entry.csv_path]
        best = ranked[0]
        second = ranked[1] if len(ranked) > 1 else None
        image_ranked = image_to_candidates[best.image_entry.image_path]
        reciprocal_best = image_ranked[0] if image_ranked else None
        csv_margin = best.score - (second.score if second is not None else 0.0)
        image_margin = best.score - (image_ranked[1].score if len(image_ranked) > 1 else 0.0)
        is_confirmed = (
            reciprocal_best is not None
            and reciprocal_best.csv_entry.csv_path == csv_entry.csv_path
            and best.score >= min_score
            and csv_margin >= min_csv_margin
            and image_margin >= min_image_margin
        )
        status = "confirmed" if is_confirmed else "needs_review"
        if is_confirmed:
            matched_image_paths.add(best.image_entry.image_path)

        row = {
            "sample_id": csv_entry.sample_id,
            "csv_path": csv_entry.csv_path.as_posix(),
            "station_name_csv": csv_entry.station_name,
            "river_csv": csv_entry.river,
            "year": csv_entry.year,
            "image_path": best.image_entry.image_path.as_posix(),
            "image_title_text": best.image_entry.title_text,
            "match_score": best.score,
            "station_similarity": best.station_similarity,
            "river_similarity": best.river_similarity,
            "combined_similarity": best.combined_similarity,
            "match_status": status,
            "csv_margin": round(csv_margin, 4),
            "image_margin": round(image_margin, 4),
            "top_candidates": [
                {
                    "image_path": candidate.image_entry.image_path.as_posix(),
                    "image_title_text": candidate.image_entry.title_text,
                    "score": candidate.score,
                }
                for candidate in ranked[:3]
            ],
        }
        rows.append(row)
        if not is_confirmed:
            unresolved.append(row)

    unmatched_images = [
        image_entry.image_path.as_posix()
        for image_entry in image_entries
        if image_entry.image_path not in matched_image_paths
    ]
    summary = {
        "csv_count": len(csv_entries),
        "image_count": len(image_entries),
        "confirmed_count": sum(1 for row in rows if row["match_status"] == "confirmed"),
        "needs_review_count": len(unresolved),
        "unmatched_image_count": len(unmatched_images),
        "unmatched_images": unmatched_images,
    }
    audit = {
        "summary": summary,
        "needs_review": unresolved,
    }
    return rows, audit


def ocr_tokens_from_result(ocr_result: list[Any] | None, y_offset: float = 0.0) -> list[OcrToken]:
    tokens: list[OcrToken] = []
    for entry in ocr_result or []:
        if len(entry) < 2:
            continue
        text = str(entry[1]).strip()
        if not text:
            continue
        points = entry[0]
        left = min(point[0] for point in points)
        right = max(point[0] for point in points)
        top = min(point[1] for point in points) + y_offset
        bottom = max(point[1] for point in points) + y_offset
        tokens.append(OcrToken(text=text, left=left, top=top, right=right, bottom=bottom))
    return tokens


def select_statistics_anchor(tokens: list[OcrToken]) -> tuple[OcrToken | None, str | None]:
    average_tokens = sorted((token for token in tokens if "平均" in token.text), key=lambda token: token.top)
    weak_average_tokens = sorted((token for token in tokens if token.text in {"平", "均"}), key=lambda token: token.top)
    year_stats = sorted((token for token in tokens if "年统计" in token.text), key=lambda token: token.top)
    statistic_labels = sorted(
        (token for token in tokens if any(label in token.text for label in ("日期", "最低", "最高", "年统计"))),
        key=lambda token: token.top,
    )

    def is_above_year_stats(token: OcrToken) -> bool:
        if not year_stats:
            return True
        return token.center_y < year_stats[0].center_y - 24.0

    def label_column_right_bound() -> float:
        candidates = [token.right for token in statistic_labels]
        if average_tokens:
            candidates.extend(token.right for token in average_tokens)
        if year_stats:
            candidates.extend(token.right for token in year_stats)
        return max(candidates, default=160.0) + 12.0

    def has_statistics_context(token: OcrToken) -> bool:
        return any(
            12.0 <= other.center_y - token.center_y <= 120.0
            for other in statistic_labels
            if other.top > token.top
        )

    if len(average_tokens) >= 2:
        if year_stats:
            anchor = next(
                (token for token in average_tokens if abs(token.center_y - year_stats[0].center_y) > 24.0),
                average_tokens[0],
            )
        else:
            anchor = average_tokens[0]
        return anchor, "average_anchor"

    if average_tokens and year_stats:
        if average_tokens[0].center_y < year_stats[0].center_y - 24.0:
            return average_tokens[0], "average_anchor"

    if average_tokens:
        return average_tokens[0], "average_anchor_fallback"

    weak_label_bound = label_column_right_bound()
    weak_candidates = [
        token
        for token in weak_average_tokens
        if token.right <= weak_label_bound
        and token.left <= weak_label_bound - 24.0
        and is_above_year_stats(token)
        and has_statistics_context(token)
    ]
    if weak_candidates:
        return weak_candidates[0], "weak_average_anchor"

    return None, None


def apply_bottom_buffer(cut_y: int, height: int, bottom_buffer_px: int = 6) -> int:
    return min(max(cut_y + bottom_buffer_px, 1), height)


def find_horizontal_cut_y(
    row_dark_counts: list[int],
    width: int,
    anchor_top: float,
    min_ratio: float = 0.6,
    search_up_px: int = 80,
    search_down_px: int = 8,
) -> int | None:
    threshold = width * min_ratio
    start = max(int(anchor_top) - search_up_px, 0)
    end = min(int(anchor_top) + search_down_px, len(row_dark_counts))
    for row_index in range(end - 1, start - 1, -1):
        if row_dark_counts[row_index] >= threshold:
            return row_index + 1
    return None


def fallback_cut_y_from_lines(
    row_dark_counts: list[int],
    width: int,
    height: int,
    min_ratio: float = 0.6,
    start_ratio: float = 0.6,
) -> int | None:
    threshold = width * min_ratio
    start = int(height * start_ratio)
    for row_index in range(start, len(row_dark_counts)):
        if row_dark_counts[row_index] >= threshold:
            return row_index + 1
    return None
