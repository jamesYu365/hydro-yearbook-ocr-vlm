from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from yearbook_ocr.common.jsonl import load_json, load_jsonl
from yearbook_ocr.eval.prediction_eval import score_sample


def sample_metrics(predictions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    payload = {}
    for row in predictions:
        result = score_sample(row["sample_id"], row["prediction"], row["target_csv"])
        payload[row["sample_id"]] = {
            "sample_id": row["sample_id"],
            "image_path": row["image_path"],
            "prediction": row["prediction"],
            "target_csv": row["target_csv"],
            "weighted_score": result.weighted_score,
            "character_accuracy": result.character_accuracy,
            "cell_accuracy": result.cell_accuracy,
            "error_type": result.error_type,
        }
    return payload


def choose_examples(
    base_by_id: dict[str, dict[str, Any]],
    ckpt_by_id: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    shared_ids = sorted(set(base_by_id) & set(ckpt_by_id))
    improved = max(
        shared_ids,
        key=lambda sample_id: ckpt_by_id[sample_id]["weighted_score"] - base_by_id[sample_id]["weighted_score"],
    )
    both_fail_candidates = [
        sample_id for sample_id in shared_ids
        if base_by_id[sample_id]["weighted_score"] < 1.0 and ckpt_by_id[sample_id]["weighted_score"] < 1.0
    ]
    if both_fail_candidates:
        both_fail = min(
            both_fail_candidates,
            key=lambda sample_id: (base_by_id[sample_id]["weighted_score"] + ckpt_by_id[sample_id]["weighted_score"]) / 2.0,
        )
    else:
        both_fail = improved
    both_success_candidates = [
        sample_id for sample_id in shared_ids
        if base_by_id[sample_id]["weighted_score"] == 1.0 and ckpt_by_id[sample_id]["weighted_score"] == 1.0
    ]
    if both_success_candidates:
        both_success = both_success_candidates[0]
    else:
        both_success = max(
            shared_ids,
            key=lambda sample_id: (base_by_id[sample_id]["weighted_score"] + ckpt_by_id[sample_id]["weighted_score"]) / 2.0,
        )
    return {
        "improved": {"sample_id": improved, "base": base_by_id[improved], "ckpt": ckpt_by_id[improved]},
        "both_fail": {"sample_id": both_fail, "base": base_by_id[both_fail], "ckpt": ckpt_by_id[both_fail]},
        "both_success": {"sample_id": both_success, "base": base_by_id[both_success], "ckpt": ckpt_by_id[both_success]},
    }


def clip_text(text: str, limit: int = 500) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n..."


def format_summary_row(label: str, summary: dict[str, Any]) -> str:
    errors = summary["by_error_type"]
    return (
        f"| {label} | {summary['weighted_score']:.4f} | {summary['character_accuracy']:.4f} | "
        f"{summary['cell_accuracy']:.4f} | {errors.get('truncation', 0)} | "
        f"{errors.get('structure_error', 0)} | {errors.get('value_error', 0)} |"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write a markdown comparison report for real GOT inference runs.")
    parser.add_argument("--base-predictions", type=Path, required=True)
    parser.add_argument("--base-eval", type=Path, required=True)
    parser.add_argument("--ckpt-predictions", type=Path, required=True)
    parser.add_argument("--ckpt-eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-label", type=str, default="base_got_ocr2")
    parser.add_argument("--ckpt-label", type=str, default="single_gpu_ckpt45")
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/flow_real_test_aligned.jsonl"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    base_predictions = load_jsonl(args.base_predictions)
    ckpt_predictions = load_jsonl(args.ckpt_predictions)
    base_eval = load_json(args.base_eval)
    ckpt_eval = load_json(args.ckpt_eval)

    base_by_id = sample_metrics(base_predictions)
    ckpt_by_id = sample_metrics(ckpt_predictions)
    examples = choose_examples(base_by_id, ckpt_by_id)

    lines = [
        "# GOT-OCR2.0 Real Daily Flow Test Report",
        "",
        "## Setup",
        "",
        f"- Manifest: `{args.manifest.as_posix()}`",
        f"- Sample count: `{len(base_predictions)}`",
        f"- Base model: `{args.base_label}`",
        f"- Trained model: `{args.ckpt_label}`",
        f"- Checkpoint: `{args.checkpoint_path}`",
        "",
        "## Summary",
        "",
        "| Model | Weighted | Char Acc | Cell Acc | Truncation | Structure | Value |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        format_summary_row(args.base_label, base_eval["summary"]),
        format_summary_row(args.ckpt_label, ckpt_eval["summary"]),
        "",
        "## Conclusion",
        "",
    ]
    base_score = base_eval["summary"]["weighted_score"]
    ckpt_score = ckpt_eval["summary"]["weighted_score"]
    if ckpt_score > base_score:
        lines.append(
            f"- `{args.ckpt_label}` outperforms `{args.base_label}` on the aligned real-flow test set by "
            f"`{ckpt_score - base_score:.4f}` weighted score."
        )
    elif ckpt_score < base_score:
        lines.append(
            f"- `{args.ckpt_label}` underperforms `{args.base_label}` by "
            f"`{base_score - ckpt_score:.4f}` weighted score on the aligned real-flow test set."
        )
    else:
        lines.append(f"- `{args.ckpt_label}` and `{args.base_label}` tie on weighted score.")
    lines.extend(["", "## Example Samples", ""])
    for title, payload in (("Checkpoint Improvement", examples["improved"]), ("Both Fail", examples["both_fail"]), ("Both Success", examples["both_success"])):
        base = payload["base"]
        ckpt = payload["ckpt"]
        lines.extend(
            [
                f"### {title}: `{payload['sample_id']}`",
                "",
                f"- Image: `{base['image_path']}`",
                f"- Base weighted score: `{base['weighted_score']:.4f}`",
                f"- Checkpoint weighted score: `{ckpt['weighted_score']:.4f}`",
                "",
                "Base prediction:",
                "```text",
                clip_text(base["prediction"]),
                "```",
                "",
                "Checkpoint prediction:",
                "```text",
                clip_text(ckpt["prediction"]),
                "```",
                "",
                "Target label:",
                "```text",
                clip_text(base["target_csv"]),
                "```",
                "",
            ]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output.as_posix())
