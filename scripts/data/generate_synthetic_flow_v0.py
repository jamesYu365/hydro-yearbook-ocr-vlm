from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from yearbook_ocr.common.jsonl import write_json as dump_json
from yearbook_ocr.common.progress import progress
from yearbook_ocr.common.tabular import (
    csv_rows_to_got_format,
    csv_rows_to_text,
    is_blank_row,
    is_valid_numeric_token,
    month_day_limit,
    parse_csv_text,
    read_csv_text,
    normalize_target_rows,
    sample_id_from_name,
    seeded_random,
)

DEFAULT_FONT_CANDIDATES = (
    Path("/usr/share/fonts/MyFonts/msyh.ttf"),
    Path("/usr/share/fonts/MyFonts/MSYH.TTC"),
    Path("/usr/share/fonts/MyFonts/MSYHBD.TTC"),
    Path("/usr/share/fonts/MyFonts/msyhbd.ttf"),
    Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
    Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
    Path("/usr/share/fonts/MyFonts/simsun.ttc"),
    Path("/usr/share/fonts/MyFonts/simsunb.ttf"),
    Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
    Path("/usr/share/fonts/truetype/arphic/uming.ttc"),
    Path("/usr/share/fonts/truetype/arphic/ukai.ttc"),
)


@dataclass(frozen=True)
class RenderConfig:
    image_width: int = 2160
    image_height: int = 1090
    margin_x: int = 60
    margin_y: int = 35
    header_height: int = 70
    row_height: int = 25
    body_font_size: int = 17


DEFAULT_RENDER_CONFIG = RenderConfig()
DATA_REGIME_WEIGHTS = (
    ("normal", 45),
    ("zero_heavy", 25),
    ("zero_with_spikes", 15),
    ("calendar_tail_focus", 15),
)
VALID_DATA_REGIMES = tuple(regime for regime, _ in DATA_REGIME_WEIGHTS)


def resolve_font_path(font_path: Path | None) -> Path:
    if font_path is not None:
        if not font_path.exists():
            raise FileNotFoundError(f"Font file not found: {font_path}")
        return font_path
    for candidate in DEFAULT_FONT_CANDIDATES:
        if candidate.exists():
            return candidate
    formatted = "\n".join(f"- {path}" for path in DEFAULT_FONT_CANDIDATES)
    raise FileNotFoundError(
        "No usable Chinese font was found. Pass --font-path explicitly or install one of:\n"
        f"{formatted}"
    )


def font_display_name(font_path: Path) -> str:
    font = ImageFont.truetype(str(font_path), 22)
    try:
        family, style = font.getname()
    except OSError:
        return font_path.name
    return f"{family} {style}".strip()


def discover_template_rows(dataset_dir: Path) -> list[list[str]]:
    first_csv = sorted(dataset_dir.glob("*.csv"))[0]
    csv_text, _ = read_csv_text(first_csv)
    return parse_csv_text(csv_text)


def build_numeric_pools(dataset_dir: Path) -> dict[int, list[str]]:
    pools: dict[int, list[str]] = defaultdict(list)
    for csv_path in sorted(dataset_dir.glob("*.csv")):
        csv_text, _ = read_csv_text(csv_path)
        rows = parse_csv_text(csv_text)
        for row in rows[1:]:
            if not row or is_blank_row(row):
                continue
            for month_index, token in enumerate(row[1:13], start=0):
                if is_valid_numeric_token(token):
                    pools[month_index].append(token)
    return dict(pools)


def select_data_regime(rng) -> str:
    draw = rng.uniform(0, sum(weight for _, weight in DATA_REGIME_WEIGHTS))
    cumulative = 0.0
    for regime, weight in DATA_REGIME_WEIGHTS:
        cumulative += weight
        if draw <= cumulative:
            return regime
    return DATA_REGIME_WEIGHTS[-1][0]


def select_data_regime_for_version(rng, dataset_version: str) -> str:
    if dataset_version == "flow_v0":
        return "normal"
    return select_data_regime(rng)


def sample_pool_value(pools: dict[int, list[str]], month_index: int, rng) -> str:
    month_pool = pools[month_index]
    return rng.choice(month_pool) if month_pool else "0"


def sample_spike_value(pools: dict[int, list[str]], month_index: int, rng) -> str:
    spike_pool = [value for value in pools.get(month_index, []) if is_valid_numeric_token(value) and float(value) >= 20]
    return rng.choice(spike_pool) if spike_pool else sample_pool_value(pools, month_index, rng)


def sample_regime_value(
    day: int,
    month_index: int,
    pools: dict[int, list[str]],
    rng,
    data_regime: str,
    zero_months: set[int],
    spike_months: set[int],
) -> str:
    if data_regime == "normal":
        value = sample_pool_value(pools, month_index, rng)
        return "" if rng.random() < 0.005 else value
    if data_regime == "zero_heavy":
        zero_probability = 0.95 if month_index in zero_months else 0.65
        return "0" if rng.random() < zero_probability else sample_pool_value(pools, month_index, rng)
    if data_regime == "zero_with_spikes":
        if month_index in zero_months:
            return sample_spike_value(pools, month_index, rng) if rng.random() < 0.08 else "0"
        if month_index in spike_months and rng.random() < 0.25:
            return sample_spike_value(pools, month_index, rng)
        return "0" if rng.random() < 0.55 else sample_pool_value(pools, month_index, rng)
    if data_regime == "calendar_tail_focus":
        if day >= 29:
            return "0" if rng.random() < 0.65 else sample_pool_value(pools, month_index, rng)
        return "0" if rng.random() < 0.20 else sample_pool_value(pools, month_index, rng)
    raise ValueError(f"Unsupported data regime: {data_regime}")


def sample_table_rows(
    template_rows: list[list[str]],
    pools: dict[int, list[str]],
    rng,
    data_regime: str = "normal",
) -> list[list[str]]:
    if data_regime not in VALID_DATA_REGIMES:
        raise ValueError(f"Unsupported data regime: {data_regime}")
    zero_month_count = rng.randint(6, 10) if data_regime in {"zero_heavy", "zero_with_spikes"} else 0
    zero_months = set(rng.sample(range(12), k=zero_month_count)) if zero_month_count else set()
    spike_months = set(rng.sample(range(12), k=rng.randint(1, 3))) if data_regime == "zero_with_spikes" else set()
    rows: list[list[str]] = []
    for row_index, row in enumerate(template_rows):
        if row_index == 0:
            rows.append(list(row))
            continue
        if is_blank_row(row):
            rows.append([""] * len(row))
            continue
        day = int(row[0])
        new_row = [row[0]]
        for month_index in range(12):
            if day > month_day_limit(month_index):
                new_row.append("")
                continue
            new_row.append(
                sample_regime_value(
                    day=day,
                    month_index=month_index,
                    pools=pools,
                    rng=rng,
                    data_regime=data_regime,
                    zero_months=zero_months,
                    spike_months=spike_months,
                )
            )
        rows.append(new_row)
    return rows


def normalize_table_header(rows: list[list[str]]) -> list[list[str]]:
    normalized = [list(row) for row in rows]
    if normalized and normalized[0]:
        normalized[0][0] = r"日\月"
    return normalized


def perturbation_count(rng) -> int:
    threshold = rng.random()
    if threshold < 0.10:
        return 0
    if threshold < 0.55:
        return 1
    if threshold < 0.90:
        return 2
    return rng.randint(3, 4)


def apply_shadow(image: Image.Image, rng) -> Image.Image:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = image.size
    left = rng.randint(0, width // 3)
    top = rng.randint(0, height // 3)
    right = min(width, left + rng.randint(width // 3, width))
    bottom = min(height, top + rng.randint(height // 3, height))
    alpha = rng.randint(18, 50)
    draw.rectangle((left, top, right, bottom), fill=(0, 0, 0, alpha))
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def apply_geometric_transform(image: Image.Image, rng) -> Image.Image:
    angle = rng.uniform(-1.8, 1.8)
    rotated = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor="white")
    shear = rng.uniform(-0.03, 0.03)
    width, height = rotated.size
    return rotated.transform(
        (width, height),
        Image.Transform.AFFINE,
        (1, shear, 0, 0, 1, 0),
        resample=Image.Resampling.BICUBIC,
        fillcolor="white",
    )


def apply_perturbations(image: Image.Image, rng) -> tuple[Image.Image, list[dict[str, Any]]]:
    perturbations = []
    candidates = ["brightness", "contrast", "blur", "shadow", "geometry"]
    chosen = rng.sample(candidates, k=min(perturbation_count(rng), len(candidates)))
    for name in chosen:
        if name == "brightness":
            factor = rng.uniform(0.82, 1.18)
            image = ImageEnhance.Brightness(image).enhance(factor)
            perturbations.append({"type": name, "strength": round(abs(factor - 1.0), 4)})
        elif name == "contrast":
            factor = rng.uniform(0.85, 1.2)
            image = ImageEnhance.Contrast(image).enhance(factor)
            perturbations.append({"type": name, "strength": round(abs(factor - 1.0), 4)})
        elif name == "blur":
            radius = rng.uniform(0.3, 1.2)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))
            perturbations.append({"type": name, "strength": round(radius, 4)})
        elif name == "shadow":
            image = apply_shadow(image, rng)
            perturbations.append({"type": name, "strength": 1.0})
        elif name == "geometry":
            image = apply_geometric_transform(image, rng)
            perturbations.append({"type": name, "strength": 1.0})
    return image, perturbations


Line = tuple[float, float, float, float]


def append_unique_line(lines: list[Line], line: Line) -> None:
    if line not in lines:
        lines.append(line)


def table_grid_lines(
    row_count: int,
    col_count: int,
    left: float,
    top: float,
    col_width: float,
    header_height: float,
    row_height: float,
) -> list[Line]:
    right = left + col_count * col_width
    header_bottom = top + header_height
    bottom = header_bottom + max(row_count - 1, 0) * row_height
    lines: list[Line] = []

    # Keep only the table top, the header separator, and month column boundaries.
    append_unique_line(lines, (left, top, right, top))
    append_unique_line(lines, (left, header_bottom, right, header_bottom))
    for col_index in range(col_count + 1):
        x = left + col_index * col_width
        append_unique_line(lines, (x, top, x, bottom))
    return lines


def draw_table_grid(draw: ImageDraw.ImageDraw, lines: list[Line]) -> None:
    for x0, y0, x1, y1 in lines:
        draw.line((x0, y0, x1, y1), fill="black", width=1)


def draw_diagonal_day_month_header(
    draw: ImageDraw.ImageDraw,
    bbox: tuple[float, float, float, float],
    font: ImageFont.FreeTypeFont,
) -> None:
    x0, y0, x1, y1 = bbox
    draw.line((x0, y0, x1, y1), fill="black", width=1)

    day = "日"
    month = "月"
    day_bbox = draw.textbbox((0, 0), day, font=font)
    month_bbox = draw.textbbox((0, 0), month, font=font)
    day_width = day_bbox[2] - day_bbox[0]
    day_height = day_bbox[3] - day_bbox[1]
    month_width = month_bbox[2] - month_bbox[0]
    month_height = month_bbox[3] - month_bbox[1]
    cell_width = x1 - x0
    cell_height = y1 - y0

    month_x = x0 + cell_width * 0.72 - month_width / 2
    month_y = y0 + cell_height * 0.20 - month_height / 2
    day_x = x0 + cell_width * 0.28 - day_width / 2
    day_y = y0 + cell_height * 0.78 - day_height / 2
    draw.text((month_x, month_y), month, fill="black", font=font)
    draw.text((day_x, day_y), day, fill="black", font=font)


def render_table(
    rows: list[list[str]],
    font_path: Path,
    rng,
    config: RenderConfig = DEFAULT_RENDER_CONFIG,
) -> tuple[Image.Image, list[dict[str, Any]]]:
    width = config.image_width
    height = config.image_height
    margin_x = config.margin_x
    margin_y = config.margin_y
    cols = len(rows[0])
    table_width = width - margin_x * 2
    col_width = table_width / cols
    header_height = config.header_height
    row_height = config.row_height
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    body_font = ImageFont.truetype(str(font_path), config.body_font_size)
    draw_table_grid(
        draw,
        table_grid_lines(
            row_count=len(rows),
            col_count=cols,
            left=margin_x,
            top=margin_y,
            col_width=col_width,
            header_height=header_height,
            row_height=row_height,
        ),
    )

    cells = []
    table_top = margin_y
    for row_index, row in enumerate(rows):
        if row_index == 0:
            y0 = table_top
            y1 = table_top + header_height
        else:
            y0 = table_top + header_height + (row_index - 1) * row_height
            y1 = y0 + row_height
        for col_index, value in enumerate(row):
            x0 = margin_x + col_index * col_width
            x1 = x0 + col_width
            text = str(value)
            if text:
                if row_index == 0 and col_index == 0:
                    draw_diagonal_day_month_header(draw, (x0, y0, x1, y1), body_font)
                else:
                    bbox = draw.textbbox((0, 0), text, font=body_font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    jitter_x = rng.uniform(-3, 3) if row_index > 0 and col_index > 0 and rng.random() < 0.15 else 0
                    jitter_y = rng.uniform(-2, 2) if row_index > 0 and col_index > 0 and rng.random() < 0.15 else 0
                    cell_height = y1 - y0
                    text_x = x0 + (col_width - text_width) / 2 + jitter_x
                    text_y = y0 + (cell_height - text_height) / 2 + jitter_y - 2
                    draw.text((text_x, text_y), text, fill="black", font=body_font)
            cell_type = "separator" if is_blank_row(row) else "data"
            if row_index == 0:
                cell_type = "header"
            elif col_index == 0 and not is_blank_row(row):
                cell_type = "index"
            cells.append(
                {
                    "row_index": row_index,
                    "col_index": col_index,
                    "text": text,
                    "bbox": [round(x0), round(y0), round(x1), round(y1)],
                    "cell_type": cell_type,
                }
            )
    return image, cells


def write_manifest_line(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_sample_payload(
    index: int,
    seed: int,
    split: str,
    dataset_version: str,
    template_rows: list[list[str]],
    pools: dict[int, list[str]],
    font_path: Path,
    font_name: str,
    render_config: RenderConfig,
    image_dir: Path,
    layout_dir: Path,
) -> tuple[str, dict[str, Any]]:
    rng = seeded_random(seed + index)
    data_regime = select_data_regime_for_version(rng, dataset_version)
    sample_rows = normalize_table_header(sample_table_rows(template_rows, pools, rng, data_regime=data_regime))
    target_rows = normalize_target_rows(sample_rows)
    sample_csv = csv_rows_to_text(target_rows)
    sample_got_format = csv_rows_to_got_format(target_rows)
    image, cells = render_table(sample_rows, font_path, rng, config=render_config)
    image, perturbations = apply_perturbations(image, rng)

    sample_name = f"{dataset_version}_{index:05d}"
    sample_id = sample_id_from_name(sample_name)
    image_path = image_dir / f"{sample_name}.png"
    layout_path = layout_dir / f"{sample_name}.json"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path)

    dump_json(
        layout_path,
        {
            "sample_id": sample_id,
            "image_path": str(image_path.as_posix()),
            "table_meta": {
                "task": dataset_version,
                "layout_name": "fixed_flow_table",
                "data_regime": data_regime,
                "row_count": len(sample_rows),
                "col_count": len(sample_rows[0]),
                "font_path": font_path.as_posix(),
                "font_name": font_name,
                **asdict(render_config),
            },
            "cells": cells,
            "perturbations": perturbations,
        },
    )

    record = {
        "sample_id": sample_id,
        "image_path": str(image_path.as_posix()),
        "target_csv": sample_csv,
        "target_got_format": sample_got_format,
        "layout_json_path": str(layout_path.as_posix()),
        "perturbations": perturbations,
        "data_regime": data_regime,
        "source_template_id": "fixed_flow_table",
        "split": split,
        "source": "synthetic",
    }
    return split, record


def build_split_map(num_samples: int, val_ratio: float, seed: int) -> dict[int, str]:
    if not 0 <= val_ratio <= 1:
        raise ValueError("--val-ratio must be between 0 and 1")
    indices = list(range(num_samples))
    rng = seeded_random(seed + 1_000_000)
    rng.shuffle(indices)
    val_count = round(num_samples * val_ratio)
    val_indices = set(indices[:val_count])
    return {index: ("val" if index in val_indices else "train") for index in range(num_samples)}


def build_manifest_record_from_existing_assets(
    index: int,
    seed: int,
    split: str,
    dataset_version: str,
    template_rows: list[list[str]],
    pools: dict[int, list[str]],
    image_dir: Path,
    layout_dir: Path,
) -> tuple[str, dict[str, Any]]:
    rng = seeded_random(seed + index)
    data_regime = select_data_regime_for_version(rng, dataset_version)
    sample_rows = normalize_table_header(sample_table_rows(template_rows, pools, rng, data_regime=data_regime))
    target_rows = normalize_target_rows(sample_rows)
    sample_csv = csv_rows_to_text(target_rows)
    sample_got_format = csv_rows_to_got_format(target_rows)

    sample_name = f"{dataset_version}_{index:05d}"
    sample_id = sample_id_from_name(sample_name)
    image_path = image_dir / f"{sample_name}.png"
    layout_path = layout_dir / f"{sample_name}.json"
    if not image_path.exists():
        raise FileNotFoundError(f"Missing generated image: {image_path}")
    if not layout_path.exists():
        raise FileNotFoundError(f"Missing generated layout: {layout_path}")
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    perturbations = layout.get("perturbations", [])
    record = {
        "sample_id": sample_id,
        "image_path": str(image_path.as_posix()),
        "target_csv": sample_csv,
        "target_got_format": sample_got_format,
        "layout_json_path": str(layout_path.as_posix()),
        "perturbations": perturbations,
        "data_regime": data_regime,
        "source_template_id": "fixed_flow_table",
        "split": split,
        "source": "synthetic",
    }
    return split, record


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic flow tables.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/流量/2006"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/flow_v0"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/manifests/flow_v0"))
    parser.add_argument("--dataset-version", type=str, default="flow_v0")
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260408)
    parser.add_argument(
        "--font-path",
        type=Path,
        default=None,
        help="Optional path to a Chinese-capable font. If omitted, the script auto-selects a system font.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker processes for sample image/layout generation. Use 1 for sequential generation.",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Rebuild train/val manifests from existing generated images and layouts without rewriting assets.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable the default progress bar.")
    args = parser.parse_args()

    if args.num_workers < 1:
        raise ValueError("--num-workers must be at least 1")
    if not 0 <= args.val_ratio <= 1:
        raise ValueError("--val-ratio must be between 0 and 1")

    font_path = resolve_font_path(args.font_path)
    font_name = font_display_name(font_path)
    render_config = DEFAULT_RENDER_CONFIG
    template_rows = discover_template_rows(args.dataset_dir)
    pools = build_numeric_pools(args.dataset_dir)

    image_dir = args.output_dir / "images"
    layout_dir = args.output_dir / "layouts"
    manifest_dir = args.manifest_dir
    train_manifest = manifest_dir / "train.jsonl"
    val_manifest = manifest_dir / "val.jsonl"
    if train_manifest.exists():
        train_manifest.unlink()
    if val_manifest.exists():
        val_manifest.unlink()
    split_map = build_split_map(args.num_samples, args.val_ratio, args.seed)

    if args.manifest_only:
        for index in progress(
            range(args.num_samples),
            total=args.num_samples,
            desc="manifest_flow_v0",
            unit="sample",
            disable=args.no_progress,
        ):
            split, record = build_manifest_record_from_existing_assets(
                index,
                args.seed,
                split_map[index],
                args.dataset_version,
                template_rows,
                pools,
                image_dir,
                layout_dir,
            )
            write_manifest_line(train_manifest if split == "train" else val_manifest, record)
        print(
            f"Wrote manifests under {manifest_dir} from existing assets under {args.output_dir}"
        )
        return

    payload_args = [
        (
            index,
            args.seed,
            split_map[index],
            args.dataset_version,
            template_rows,
            pools,
            font_path,
            font_name,
            render_config,
            image_dir,
            layout_dir,
        )
        for index in range(args.num_samples)
    ]
    if args.num_workers == 1:
        payload_iter = (build_sample_payload(*payload) for payload in payload_args)
    else:
        max_workers = min(args.num_workers, os.cpu_count() or args.num_workers)
        executor = ProcessPoolExecutor(max_workers=max_workers)
        payload_iter = executor.map(build_sample_payload_from_tuple, payload_args, chunksize=16)

    try:
        for split, record in progress(
            payload_iter,
            total=args.num_samples,
            desc="generate_flow_v0",
            unit="sample",
            disable=args.no_progress,
        ):
            write_manifest_line(train_manifest if split == "train" else val_manifest, record)
    finally:
        if args.num_workers != 1:
            executor.shutdown()

    print(
        f"Generated {args.num_samples} synthetic samples under {args.output_dir} "
        f"and manifests under {manifest_dir} using font {font_path}"
    )


def build_sample_payload_from_tuple(args: tuple[Any, ...]) -> tuple[str, dict[str, Any]]:
    return build_sample_payload(*args)


if __name__ == "__main__":
    main()
