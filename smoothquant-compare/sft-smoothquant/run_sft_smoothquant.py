#!/usr/bin/env python3
"""
End-to-end helper: SFT a model, then run SmoothQuant calibration + PPL eval
for multiple alpha values.

Steps:
1) Fine-tune with wiki-qwen3-sft pipeline (FP32/LoRA off by default).
2) Calibrate activation scales via SmoothQuant's generate_act_scales.py.
3) For each alpha in {0.1, 0.3, 0.5, 0.7, 0.9}, run SmoothQuant PPL eval
   (W8A8 by default) on in-domain and unseen sets.

Paths/defaults are set for the local workspace; adjust as needed.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import subprocess
from pathlib import Path
import sys
import json
import math
from typing import Optional

import torch
from torch.utils.data import DataLoader, TensorDataset

# Ensure smoothquant is importable when running directly
REPO_SQ = Path("../smoothquant")
if str(REPO_SQ) not in sys.path:
    sys.path.insert(0, str(REPO_SQ))

from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import quantize_model
from transformers import AutoModelForCausalLM


def run_cmd(cmd, cwd, env=None):
    print(f"[CMD] {' '.join(cmd)}")
    ret = subprocess.run(cmd, cwd=cwd, env=env)
    if ret.returncode != 0:
        raise RuntimeError(f"Command failed with code {ret.returncode}: {' '.join(cmd)}")


def run_cmd_capture(cmd, cwd, env=None) -> str:
    """Run a command and return combined stdout/stderr (also prints to console)."""
    print(f"[CMD] {' '.join(cmd)}")
    ret = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(ret.stdout, end="")
    if ret.returncode != 0:
        raise RuntimeError(f"Command failed with code {ret.returncode}: {' '.join(cmd)}")
    return ret.stdout


def parse_ppl(output: str) -> Optional[float]:
    val: Optional[float] = None
    for line in output.splitlines():
        if "Perplexity:" in line:
            try:
                val = float(line.split("Perplexity:")[1].strip())
            except Exception:
                continue
    return val


@torch.no_grad()
def ppl_on_blocks(model, blocks: torch.Tensor, device: str, batch_size: int) -> float:
    loader = DataLoader(TensorDataset(blocks), batch_size=batch_size, shuffle=False)
    model.eval()
    tot_nll, tot_tok = 0.0, 0
    for (x,) in loader:
        x = x.to(device)
        out = model(input_ids=x, labels=x)
        tot_nll += float(out.loss) * x.numel()
        tot_tok += x.numel()
    return math.exp(tot_nll / max(1, tot_tok))


def main():
    print(os.getcwd())
    repo_sft = Path("/workspace/smoothquant-compare/wiki-qwen3-sft")
    repo_sq = Path("/workspace/smoothquant-compare/sft-smoothquant/smoothquant")
    out_root = repo_sft.parent / "sft-smoothquant" / "outputs"
    out_root.mkdir(parents=True, exist_ok=True)
    env_sq = os.environ.copy()
    env_sq["PYTHONPATH"] = f"{repo_sq}:{env_sq.get('PYTHONPATH', '')}"

    # Pin SFT (first stage) and SmoothQuant to the same GPU (change if needed)
    env_sft = env_sq.copy()
    env_sft["CUDA_VISIBLE_DEVICES"] = "0"
    env_sq["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # ----- Config -----
    model_path = "facebook/opt-1.3b"
    base_tokenizer_path = Path(model_path)  # assumes local; adjust if needed
    dataset_in = "wikitext:wikitext-2-raw-v1"
    dataset_unseen = "lambada:plain_text"

    # SFT config
    steps = "0"
    block_size = "256"
    batch_size_ft = "4"
    weight_bits = "4"
    act_bits = "4"

    # SmoothQuant config
    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]  # multiple alphas to sweep
    calib_samples = "512"
    calib_seq_len = block_size

    tag = f"sft_sq_steps{steps}_bs{block_size}_ft{batch_size_ft}_w{weight_bits}a{act_bits}"
    sft_out = out_root / f"{tag}_sft"
    sq_scales = out_root / f"{tag}_act_scales.json"  # actually torch.save'd; extension kept for consistency
    calib_jsonl = out_root / f"{tag}_calib.jsonl"
    sq_n_samples_in: Optional[int] = None
    sq_n_samples_un: Optional[int] = None

    # ----- 1) SFT (run ONCE) -----
    run_cmd(
        [
            "python",
            str(repo_sft / "scripts" / "next_token" / "core_alg" / "quant_sft_pipeline_fakequant.py"),
            "--model",
            str(model_path),
            "--dataset-in",
            dataset_in,
            "--dataset-unseen",
            dataset_unseen,
            "--steps",
            steps,
            "--block-size",
            block_size,
            "--batch-size-ft",
            batch_size_ft,
            "--weight-bits",
            weight_bits,
            "--act-bits",
            act_bits,
            "--eval-every",
            "0",
            "--lambda-x",
            "0",
            "--output-dir",
            str(sft_out),
            "--device",
            "cuda:0",
            # "--use-qat",
            # "--qat-bits-w",
            # "4",
        ],
        cwd=repo_sft,
        env=env_sft,
    )


    # Load SFT eval block counts to reuse the same eval subset for SQ
    results_path = sft_out / "results.json"
    if results_path.exists():
        try:
            sft_results = json.loads(results_path.read_text())
            sq_n_samples_in = int(sft_results.get("nblocks_in", 0) or 0) or None
            sq_n_samples_un = int(sft_results.get("nblocks_unseen", 0) or 0) or None
            print(f"[INFO] SFT eval subset sizes: in={sq_n_samples_in}, unseen={sq_n_samples_un}")
        except Exception as e:
            print(f"[WARN] Failed to read SFT results for SQ eval subset: {e}")

    # ----- 2) SmoothQuant calibration (run ONCE) -----
    # Ensure tokenizer files exist in the SFT output dir (needed for from_pretrained).
    for fname in ["tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"]:
        src = base_tokenizer_path / fname
        dst = sft_out / fname
        if src.exists() and not dst.exists():
            dst.write_bytes(src.read_bytes())

    run_cmd(
        [
            "python",
            str(repo_sq / "examples" / "generate_act_scales.py"),
            "--model-name",
            str(sft_out),
            "--output-path",
            str(sq_scales),
            "--num-samples",
            calib_samples,
            "--seq-len",
            calib_seq_len,
            "--dataset-path",
            str(calib_jsonl),
            "--dataset-name",
            "wikitext",
            "--dataset-subset",
            "wikitext-2-raw-v1",
            "--dataset-split",
            "test",
            "--device-map",
            "cuda:0",
        ],
        cwd=repo_sq,
        env=env_sq,
    )

    # ----- 3) Load eval blocks (reused for all alphas) -----
    blocks_in_path = sft_out / "eval_blocks_in.pt"
    blocks_un_path = sft_out / "eval_blocks_unseen.pt"
    if not blocks_in_path.exists():
        raise FileNotFoundError(f"Missing eval blocks: {blocks_in_path}")
    data_in = torch.load(blocks_in_path, map_location="cpu")
    blocks_in = data_in["blocks"]
    batch_eval_in = int(data_in.get("batch_size_eval", 1))

    blocks_un = None
    batch_eval_un = 1
    if blocks_un_path.exists():
        data_un = torch.load(blocks_un_path, map_location="cpu")
        blocks_un = data_un["blocks"]
        batch_eval_un = int(data_un.get("batch_size_eval", 1))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        print(f"[INFO] SQ eval visible devices={visible}, using device {device}")

    # Load act scales (shared across alphas)
    act_scales = torch.load(sq_scales, map_location="cpu")

    # To collect results for a final summary
    alpha_results = []

    # ----- 4) Sweep over multiple alphas -----
    for alpha in alphas:
        alpha_str = f"{alpha:.1f}".replace(".", "p")
        sq_report = out_root / f"{tag}_sq_alpha{alpha_str}.txt"

        print(f"\n[INFO] Evaluating SmoothQuant with alpha={alpha}...\n")

        with open(sq_report, "w", encoding="utf-8") as f:
            f.write(f"SFT dir: {sft_out}\n")
            f.write(f"Act scales: {sq_scales}\n")
            f.write(f"Alpha: {alpha}\n\n")

        # Fresh model for each alpha
        model = AutoModelForCausalLM.from_pretrained(
            sft_out,
            torch_dtype=torch.bfloat16,
        ).to(device)

        # Move act scales to device (structure-dependent, so we just move tensors)
        act_scales_dev = {}
        for k, v in act_scales.items():
            if isinstance(v, torch.Tensor):
                act_scales_dev[k] = v.to(device)
            else:
                act_scales_dev[k] = v

        # Apply SmoothQuant + fake quant
        smooth_lm(model, act_scales_dev, float(alpha))
        model = quantize_model(
            model,
            weight_quant="per_channel",
            act_quant="per_token",
            quantize_bmm_input=True,
            n_bits=int(weight_bits),
        )

        # PPL on in-domain
        sq_ppl_in = ppl_on_blocks(model, blocks_in, device=device, batch_size=batch_eval_in)
        # PPL on unseen
        sq_ppl_un = None
        if blocks_un is not None:
            sq_ppl_un = ppl_on_blocks(model, blocks_un, device=device, batch_size=batch_eval_un)
        if sq_ppl_un is None:
            sq_ppl_un = float("nan")

        # Print result for this alpha
        print(
            f"[RESULT] alpha={alpha:.1f} | "
            f"in-domain PPL={sq_ppl_in:.4f} | "
            f"unseen PPL={sq_ppl_un:.4f}"
        )

        with open(sq_report, "a", encoding="utf-8") as f:
            f.write(f"SmoothQuant PPL (in-domain): {sq_ppl_in}\n")
            f.write(f"SmoothQuant PPL (unseen): {sq_ppl_un}\n")

        # Append to in-memory summary
        alpha_results.append(
            {
                "alpha": float(alpha),
                "in_domain_ppl": float(sq_ppl_in),
                "unseen_ppl": float(sq_ppl_un),
            }
        )

        # Append/update SQ eval results into SFT results.json
        if results_path.exists():
            try:
                data = json.loads(results_path.read_text())

                # Keep legacy single-value fields as "last alpha run" (optional)
                if "in_domain" in data:
                    data["in_domain"]["smoothquant_ppl"] = sq_ppl_in
                if "unseen" in data:
                    data["unseen"]["smoothquant_ppl"] = sq_ppl_un

                # Store per-alpha results under "smoothquant"
                sq_dict = data.get("smoothquant", {})
                sq_dict[str(alpha)] = {
                    "alpha": float(alpha),
                    "weight_bits": int(weight_bits),
                    "act_bits": int(act_bits),
                    "in_domain_ppl": sq_ppl_in,
                    "unseen_ppl": sq_ppl_un,
                }
                data["smoothquant"] = sq_dict

                results_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
                print(f"[INFO] Updated results with SmoothQuant evals for alpha={alpha} at {results_path}")
            except Exception as e:
                print(f"[WARN] Failed to update results.json with SQ metrics for alpha={alpha}: {e}")

        # Free up GPU memory before next alpha
        del model
        del act_scales_dev
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ----- 5) Final summary print -----
    if alpha_results:
        print("\n========== SmoothQuant Alpha Sweep Summary ==========")
        print(f"{'alpha':>8} | {'PPL_in':>12} | {'PPL_unseen':>12}")
        print("-" * 40)
        for res in alpha_results:
            a = res["alpha"]
            pin = res["in_domain_ppl"]
            pun = res["unseen_ppl"]
            print(f"{a:8.1f} | {pin:12.4f} | {pun:12.4f}")
        print("=====================================================\n")

    print(f"[DONE] Outputs in {out_root}")


if __name__ == "__main__":
    main()
