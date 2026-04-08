from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.common.yearbook_flow_common import (
    csv_rows_to_text,
    dump_json,
    is_blank_row,
    is_valid_numeric_token,
    month_day_limit,
    parse_csv_text,
    read_csv_text,
    sample_id_from_name,
    seeded_random,
)


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


def sample_table_rows(template_rows: list[list[str]], pools: dict[int, list[str]], rng) -> list[list[str]]:
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
            month_pool = pools[month_index]
            value = rng.choice(month_pool) if month_pool else "0"
            if rng.random() < 0.005:
                value = ""
            new_row.append(value)
        rows.append(new_row)
    return rows


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


def render_table(rows: list[list[str]], font_path: Path, rng) -> tuple[Image.Image, list[dict[str, Any]]]:
    width = 1800
    height = 2200
    margin_x = 90
    margin_y = 110
    cols = len(rows[0])
    table_width = width - margin_x * 2
    col_width = table_width / cols
    row_height = 46
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.truetype(str(font_path), 30)
    body_font = ImageFont.truetype(str(font_path), 22)
    draw.text((margin_x, 40), "流量表", fill="black", font=title_font)

    cells = []
    table_top = margin_y
    for row_index, row in enumerate(rows):
        y0 = table_top + row_index * row_height
        y1 = y0 + row_height
        for col_index, value in enumerate(row):
            x0 = margin_x + col_index * col_width
            x1 = x0 + col_width
            draw.rectangle((x0, y0, x1, y1), outline="black", width=1)
            text = str(value)
            if text:
                bbox = draw.textbbox((0, 0), text, font=body_font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                jitter_x = rng.uniform(-3, 3) if row_index > 0 and col_index > 0 and rng.random() < 0.15 else 0
                jitter_y = rng.uniform(-2, 2) if row_index > 0 and col_index > 0 and rng.random() < 0.15 else 0
                text_x = x0 + (col_width - text_width) / 2 + jitter_x
                text_y = y0 + (row_height - text_height) / 2 + jitter_y - 2
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate v0 synthetic flow tables.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/流量/2006"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/flow_v0"))
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260408)
    parser.add_argument(
        "--font-path",
        type=Path,
        default=Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    args = parser.parse_args()

    rng = seeded_random(args.seed)
    template_rows = discover_template_rows(args.dataset_dir)
    pools = build_numeric_pools(args.dataset_dir)

    image_dir = args.output_dir / "images"
    layout_dir = args.output_dir / "layouts"
    manifest_dir = Path("data/manifests/flow_v0")
    train_manifest = manifest_dir / "train.jsonl"
    val_manifest = manifest_dir / "val.jsonl"
    if train_manifest.exists():
        train_manifest.unlink()
    if val_manifest.exists():
        val_manifest.unlink()

    for index in range(args.num_samples):
        sample_rows = sample_table_rows(template_rows, pools, rng)
        sample_csv = csv_rows_to_text(sample_rows)
        image, cells = render_table(sample_rows, args.font_path, rng)
        image, perturbations = apply_perturbations(image, rng)

        sample_name = f"flow_v0_{index:05d}"
        sample_id = sample_id_from_name(sample_name)
        split = "val" if rng.random() < args.val_ratio else "train"
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
                    "task": "flow_v0",
                    "layout_name": "fixed_flow_table",
                    "row_count": len(sample_rows),
                    "col_count": len(sample_rows[0]),
                },
                "cells": cells,
                "perturbations": perturbations,
            },
        )

        record = {
            "sample_id": sample_id,
            "image_path": str(image_path.as_posix()),
            "target_csv": sample_csv,
            "layout_json_path": str(layout_path.as_posix()),
            "perturbations": perturbations,
            "source_template_id": "fixed_flow_table",
            "split": split,
            "source": "synthetic",
        }
        write_manifest_line(train_manifest if split == "train" else val_manifest, record)

    print(
        f"Generated {args.num_samples} synthetic samples under {args.output_dir} "
        f"and manifests under {manifest_dir}"
    )


if __name__ == "__main__":
    main()
