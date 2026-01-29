import torch
import os
import sys
from pathlib import Path
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# --- Add SmoothQuant repo to sys.path ---
SMOOTHQUANT_DIR = "/workspace/smoothquant-compare/sft-smoothquant"
if SMOOTHQUANT_DIR not in sys.path:
    sys.path.insert(0, SMOOTHQUANT_DIR)

from smoothquant.calibration import get_act_scales


def build_model_and_tokenizer(model_name, device_map="cpu"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
    # If staying on CPU, keep full precision to avoid slow fp16 kernels on CPU.
    torch_dtype = torch.float16 if device_map != "cpu" else torch.float32
    kwargs = {"torch_dtype": torch_dtype, "device_map": device_map}
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return model, tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name", type=str, default="facebook/opt-1.3b", help="model name"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="act_scales/opt-1.3b.pt",
        help="where to save the act scales",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="dataset/val.jsonl.zst",
        help=(
            "Path to calibration dataset (jsonl). If file not found and "
            "--dataset-name is provided, will load from HF."
        ),
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Optional Hugging Face dataset name (e.g., wikitext) to use if dataset-path is not a local file.",
    )
    parser.add_argument(
        "--dataset-subset",
        type=str,
        default=None,
        help="Optional subset name for the HF dataset (e.g., wikitext-2-raw-v1).",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="validation",
        help="Split to use when loading HF dataset.",
    )
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument(
        "--device-map",
        type=str,
        default="cuda:0",
        help=(
            "Device map for loading model (e.g., cuda:7, auto, cpu, sequential). "
            "Default cuda:7 to target GPU 7 (override via CLI)."
        ),
    )
    args = parser.parse_args()
    return args


@torch.no_grad()
def main():
    args = parse_args()
    model, tokenizer = build_model_and_tokenizer(args.model_name, device_map=args.device_map)

    ds_path = Path(args.dataset_path)
    if not ds_path.exists():
        if args.dataset_name is None:
            raise FileNotFoundError(
                f"Cannot find the dataset at {ds_path}. "
                "Provide --dataset-name/--dataset-subset to load from Hugging Face."
            )
        # Load from HF and dump to jsonl for get_act_scales
        print(f"[INFO] Loading dataset {args.dataset_name} ({args.dataset_subset}) split={args.dataset_split} from HF")
        ds = load_dataset(args.dataset_name, args.dataset_subset, split=args.dataset_split)
        ds_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Saving calibration data to {ds_path}")
        ds.to_json(str(ds_path))

    act_scales = get_act_scales(
        model, tokenizer, args.dataset_path, args.num_samples, args.seq_len
    )

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(act_scales, args.output_path)


if __name__ == "__main__":
    main()
