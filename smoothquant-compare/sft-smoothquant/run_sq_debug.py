#!/usr/bin/env python3
"""
Debug SmoothQuant end-to-end on the base OPT-1.3B (no SFT, no training eval).

Steps:
1) Generate activation scales with wikitext-2 (HF) as calibration.
2) Run SmoothQuant PPL eval (W4A4 by default) on in-domain (wikitext-2 test) and unseen (lambada validation).

Outputs go to: sft-smoothquant/outputs/sq_debug_*
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path


def run_cmd(cmd, cwd, env=None):
    print(f"[CMD] {' '.join(cmd)}")
    ret = subprocess.run(cmd, cwd=cwd, env=env)
    if ret.returncode != 0:
        raise RuntimeError(f"Command failed with code {ret.returncode}: {' '.join(cmd)}")


def main():
    repo_sq = Path("/home/chonghej/opt_quant_smooth/smoothquant")
    model_path = Path("/home/chonghej/opt_quant_smooth/wiki-qwen3-sft/models/opt-1.3b")
    out_root = Path("/home/chonghej/opt_quant_smooth/sft-smoothquant/outputs")
    out_root.mkdir(parents=True, exist_ok=True)

    tag = "sq_debug"
    calib_jsonl = out_root / f"{tag}_calib.jsonl"
    act_scales = out_root / f"{tag}_act_scales.json"
    alpha = "0.8"
    n_bits = "4"

    env_sq = os.environ.copy()
    env_sq["PYTHONPATH"] = f"{repo_sq}:{env_sq.get('PYTHONPATH','')}"

    # 1) Calibration (auto-download wikitext-2 if not cached)
    run_cmd(
        [
            "python",
            str(repo_sq / "examples" / "generate_act_scales.py"),
            "--model-name",
            str(model_path),
            "--output-path",
            str(act_scales),
            "--num-samples",
            "512",
            "--seq-len",
            "256",
            "--dataset-path",
            str(calib_jsonl),
            "--dataset-name",
            "wikitext",
            "--dataset-subset",
            "wikitext-2-raw-v1",
            "--dataset-split",
            "test",
        ],
        cwd=repo_sq,
        env=env_sq,
    )

    # 2) SmoothQuant PPL eval (in-domain: wikitext test)
    run_cmd(
        [
            "python",
            str(repo_sq / "smoothquant" / "ppl_eval.py"),
            "--model_path",
            str(model_path),
            "--act_scales_path",
            str(act_scales),
            "--smooth",
            "--alpha",
            alpha,
            "--quantize",
            "--n-bits",
            n_bits,
        ],
        cwd=repo_sq,
        env=env_sq,
    )

    # 3) SmoothQuant PPL eval (unseen: lambada validation)
    run_cmd(
        [
            "python",
            str(repo_sq / "smoothquant" / "ppl_eval.py"),
            "--model_path",
            str(model_path),
            "--act_scales_path",
            str(act_scales),
            "--smooth",
            "--alpha",
            alpha,
            "--quantize",
            "--n-bits",
            n_bits,
            "--dataset",
            "lambada",
            "--subset",
            "plain_text",
            "--split",
            "validation",
        ],
        cwd=repo_sq,
        env=env_sq,
    )

    print(f"[DONE] SmoothQuant debug outputs in {out_root}")


if __name__ == "__main__":
    main()
