#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from enum import auto, Enum
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, StoppingCriteria

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from scripts.common.yearbook_flow_common import DEFAULT_PROMPT


DEFAULT_MANIFEST = Path("data/manifests/flow_real_test_aligned.jsonl")
DEFAULT_BASE_MODEL = Path("outputs/cache/modelscope/models/stepfun-ai/GOT-OCR2_0")
DEFAULT_CACHE_ROOT = Path("outputs/cache")
DEFAULT_SYSTEM = "        You should follow the instructions carefully and explain your answers in detail."
DEFAULT_SINGLE_QUERY = "OCR with format: "


class SeparatorStyle(Enum):
    SINGLE = auto()
    TWO = auto()
    MPT = auto()


@dataclasses.dataclass
class Conversation:
    system: str
    roles: list[str]
    messages: list[list[str | None]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "<|im_end|>"
    sep2: str | None = None
    version: str = "Unknown"
    skip_next: bool = False

    def get_prompt(self) -> str:
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep + "\n"
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        if self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        if self.sep_style == SeparatorStyle.MPT:
            ret = self.system + self.sep if self.system else ""
            for role, message in self.messages:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role: str, message: str | None) -> None:
        self.messages.append([role, message])

    def copy(self) -> "Conversation":
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version,
        )


class KeywordsStoppingCriteria(StoppingCriteria):

    def __init__(self, keywords: list[str], tokenizer: Any, input_ids: torch.Tensor) -> None:
        self.keywords = keywords
        self.keyword_ids = [tokenizer(keyword).input_ids for keyword in keywords]
        self.keyword_ids = [
            keyword_id[0] for keyword_id in self.keyword_ids if isinstance(keyword_id, list) and len(keyword_id) == 1
        ]
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
            return False
        for keyword_id in self.keyword_ids:
            if output_ids[0, -1] == keyword_id:
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
        return any(keyword in outputs for keyword in self.keywords)


class GOTImageEvalProcessor:

    def __init__(self, image_size: int = 1024) -> None:
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image)


def set_cache_env(cache_root: Path) -> None:
    hf_home = cache_root / "hf_home"
    mpl_dir = cache_root / "matplotlib"
    home_dir = cache_root / "home"
    hf_home.mkdir(parents=True, exist_ok=True)
    mpl_dir.mkdir(parents=True, exist_ok=True)
    home_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", hf_home.resolve().as_posix())
    os.environ.setdefault("TRANSFORMERS_CACHE", (hf_home / "hub").resolve().as_posix())
    os.environ.setdefault("MPLCONFIGDIR", mpl_dir.resolve().as_posix())
    os.environ.setdefault("HOME", home_dir.resolve().as_posix())


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def resolve_dtype(dtype: str, device: str) -> torch.dtype:
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    if device.startswith("cuda"):
        return torch.float16
    return torch.float32


def build_lora_config(adapter_dir: Path) -> Any:
    from peft import LoraConfig

    config_path = adapter_dir / "adapter_config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    return LoraConfig(
        r=payload["r"],
        lora_alpha=payload["lora_alpha"],
        lora_dropout=payload["lora_dropout"],
        bias=payload["bias"],
        task_type=payload["task_type"],
        target_modules=target_modules,
        inference_mode=True,
    )


def load_lora_weights(model: Any, adapter_dir: Path) -> None:
    from safetensors.torch import load_file

    state_dict = load_file((adapter_dir / "adapter_model.safetensors").as_posix())
    renamed_state = {}
    for key, value in state_dict.items():
        key = key.replace(".lora_A.weight", ".lora_A.default.weight")
        key = key.replace(".lora_B.weight", ".lora_B.default.weight")
        renamed_state[key] = value
    missing, unexpected = model.load_state_dict(renamed_state, strict=False)
    missing = [item for item in missing if ".lora_" in item]
    if missing or unexpected:
        raise RuntimeError(
            f"LoRA state_dict mismatch. missing={missing[:5]} unexpected={unexpected[:5]}"
        )


def load_model(
    base_model: Path,
    adapter_dir: Path | None,
    device: str,
    dtype: torch.dtype,
) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(base_model.as_posix(), trust_remote_code=True)
    device_map = device if device != "cpu" else "cpu"
    model = AutoModel.from_pretrained(
        base_model.as_posix(),
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map=device_map,
        torch_dtype=dtype,
        use_safetensors=True,
        pad_token_id=151643,
    ).eval()

    if adapter_dir is not None:
        from peft import get_peft_model

        lora_config = build_lora_config(adapter_dir)
        model = get_peft_model(model, lora_config)
        load_lora_weights(model, adapter_dir)
        model.eval()

    return model, tokenizer


def build_conversation_prompt(query: str) -> tuple[str, str]:
    image_tag = "<img>" + "<imgpad>" * 256 + "</img>\n"
    conv_mpt = Conversation(
        system=f"<|im_start|>system\n{DEFAULT_SYSTEM}",
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        version="mpt",
        messages=[],
        offset=0,
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>",
    )
    conv = conv_mpt.copy()
    conv.append_message(conv.roles[0], image_tag + query)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt(), conv.sep


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def clean_prediction(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def infer_one(
    model: Any,
    tokenizer: Any,
    image_processor: GOTImageEvalProcessor,
    query: str,
    image_path: Path,
    max_new_tokens: int,
) -> str:
    image = Image.open(image_path).convert("RGB")
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    image_tensor = image_processor(image)[None].to(model_device).to(model_dtype)
    prompt, stop_str = build_conversation_prompt(query)
    inputs = tokenizer([prompt], return_tensors="pt")
    input_ids = inputs["input_ids"].to(model_device)
    attention_mask = inputs["attention_mask"].to(model_device)
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
    autocast_dtype = torch.bfloat16 if model_device.type == "cuda" else None
    with torch.no_grad():
        if autocast_dtype is not None:
            with torch.autocast("cuda", dtype=autocast_dtype):
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=[image_tensor],
                    do_sample=False,
                    num_beams=1,
                    no_repeat_ngram_size=20,
                    max_new_tokens=max_new_tokens,
                    stopping_criteria=[stopping_criteria],
                )
        else:
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=[image_tensor],
                do_sample=False,
                num_beams=1,
                no_repeat_ngram_size=20,
                max_new_tokens=max_new_tokens,
                stopping_criteria=[stopping_criteria],
            )
    generated = tokenizer.decode(output_ids[0, input_ids.shape[1] :]).strip()
    if generated.endswith(stop_str):
        generated = generated[:-len(stop_str)]
    return clean_prediction(generated.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GOT-OCR2.0 on the aligned real flow daily test set.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-label", type=str, required=True)
    parser.add_argument("--query", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="auto", choices=("auto", "float16", "bfloat16", "float32"))
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    args = parser.parse_args()

    set_cache_env(args.cache_root)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    rows = load_manifest(args.manifest)
    if args.limit is not None:
        rows = rows[: args.limit]

    start_time = time.time()
    model, tokenizer = load_model(args.base_model, args.adapter_dir, device, dtype)
    image_processor = GOTImageEvalProcessor(image_size=1024)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            image_path = Path(row["image_path"])
            prediction = infer_one(
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                query=args.query,
                image_path=image_path,
                max_new_tokens=args.max_new_tokens,
            )
            payload = {
                "sample_id": row["sample_id"],
                "image_path": row["image_path"],
                "target_csv": row["target_csv"],
                "prediction": prediction,
                "model_label": args.model_label,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            print(
                json.dumps(
                    {
                        "index": index,
                        "count": len(rows),
                        "sample_id": row["sample_id"],
                        "image_path": row["image_path"],
                    },
                    ensure_ascii=False,
                )
            )

    elapsed = time.time() - start_time
    print(
        json.dumps(
            {
                "model_label": args.model_label,
                "count": len(rows),
                "device": device,
                "dtype": str(dtype),
                "elapsed_sec": round(elapsed, 3),
                "output": args.output.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
