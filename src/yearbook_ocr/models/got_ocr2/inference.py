from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from enum import auto, Enum
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, StoppingCriteria

from yearbook_ocr.common.jsonl import load_jsonl, write_json, write_jsonl
from yearbook_ocr.common.tabular import DEFAULT_GOT_FORMAT_PROMPT, DEFAULT_PROMPT

DEFAULT_MANIFEST = Path("data/manifests/flow_real_test_aligned.jsonl")
DEFAULT_BASE_MODEL = Path("outputs/cache/modelscope/models/stepfun-ai/GOT-OCR2_0")
DEFAULT_BASE_OUTPUT_DIR = Path("outputs/got_ocr2_base")
DEFAULT_CACHE_ROOT = Path("outputs/cache")
DEFAULT_SYSTEM = "        You should follow the instructions carefully and explain your answers in detail."


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
        self.keyword_ids = [keyword_id[0] for keyword_id in self.keyword_ids if isinstance(keyword_id, list) and len(keyword_id) == 1]
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
    import json

    payload = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    return LoraConfig(
        r=payload["r"],
        lora_alpha=payload["lora_alpha"],
        lora_dropout=payload["lora_dropout"],
        bias=payload["bias"],
        task_type=payload["task_type"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
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
        raise RuntimeError(f"LoRA state_dict mismatch. missing={missing[:5]} unexpected={unexpected[:5]}")


def load_model(base_model: Path, adapter_dir: Path | None, device: str, dtype: torch.dtype) -> tuple[Any, Any]:
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

        model = get_peft_model(model, build_lora_config(adapter_dir))
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


def clean_prediction(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def infer_one_local(
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
    generated = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    if generated.endswith(stop_str):
        generated = generated[:-len(stop_str)]
    return clean_prediction(generated.strip())


def infer_one_official(model: Any, tokenizer: Any, image_path: Path) -> str:
    chat_fn = getattr(model, "chat", None)
    if chat_fn is None and hasattr(model, "base_model"):
        chat_fn = getattr(model.base_model, "chat", None)
    if chat_fn is None:
        raise RuntimeError("Loaded model does not expose chat().")
    return chat_fn(tokenizer, image_path.as_posix(), "format")


def resolve_query(query_mode: str, query: str | None) -> str:
    if query is not None:
        return query
    if query_mode == "official_format":
        return DEFAULT_GOT_FORMAT_PROMPT
    return DEFAULT_PROMPT


def default_output_path(
    adapter_dir: Path | None,
    backend: str,
    single_image: bool,
    image_path: Path | None,
    limit: int | None,
) -> Path:
    root = adapter_dir if adapter_dir is not None else DEFAULT_BASE_OUTPUT_DIR
    if single_image and image_path is not None:
        return root / f"{image_path.stem}_{backend}.json"
    limit_suffix = f"first{limit}" if limit is not None else "all"
    return root / f"flow_real_{limit_suffix}_{backend}.jsonl"


def build_single_record(image_path: Path) -> dict[str, Any]:
    return {
        "sample_id": image_path.stem,
        "image_path": image_path.as_posix(),
    }


def run_inference(args: argparse.Namespace) -> Path:
    set_cache_env(args.cache_root)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    query = resolve_query(args.query_mode, args.query)

    if args.image is not None:
        rows = [build_single_record(args.image)]
        single_image = True
    else:
        rows = load_jsonl(args.manifest)
        if args.limit is not None:
            rows = rows[: args.limit]
        single_image = False

    output_path = args.output or default_output_path(
        args.adapter_dir,
        args.backend,
        single_image,
        args.image,
        args.limit,
    )

    start_time = time.time()
    model, tokenizer = load_model(args.base_model, args.adapter_dir, device, dtype)
    image_processor = GOTImageEvalProcessor(image_size=1024)
    records: list[dict[str, Any]] = []

    for index, row in enumerate(rows, start=1):
        image_path = Path(row["image_path"])
        if args.backend == "official_chat":
            prediction = infer_one_official(model, tokenizer, image_path)
        else:
            prediction = infer_one_local(
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                query=query,
                image_path=image_path,
                max_new_tokens=args.max_new_tokens,
            )
        payload = {
            "index": index,
            "sample_id": row["sample_id"],
            "image_path": row["image_path"],
            "prediction": prediction,
            "backend": args.backend,
            "query_mode": args.query_mode,
            "query": query,
            "device": device,
            "dtype": str(dtype),
        }
        if "target_csv" in row:
            payload["target_csv"] = row["target_csv"]
        if "target_got_format" in row:
            payload["target_got_format"] = row["target_got_format"]
        records.append(payload)
        print(json.dumps({"index": index, "sample_id": row["sample_id"]}, ensure_ascii=False))

    elapsed = round(time.time() - start_time, 3)
    if single_image:
        payload = records[0]
        payload["elapsed_sec"] = elapsed
        write_json(output_path, payload)
    else:
        write_jsonl(output_path, records)
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified GOT-OCR2.0 inference entrypoint.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Run inference on a single image.")
    source.add_argument("--manifest", type=Path, help="Run inference on a manifest jsonl.")
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--backend", choices=("official_chat", "local_generate"), default="official_chat")
    parser.add_argument("--query-mode", choices=("official_format", "custom_default"), default="official_format")
    parser.add_argument("--query", type=str)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="auto", choices=("auto", "float16", "bfloat16", "float32"))
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_path = run_inference(args)
    print(output_path.as_posix())
