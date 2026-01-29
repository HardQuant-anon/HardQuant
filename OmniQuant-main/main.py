#!/usr/bin/env python3
import os
import sys
import random
import copy
import gc
import json
import numpy as np
from models.LMClass import LMClass
import torch
import time
from datautils import get_loaders
from lm_eval import evaluator
from pprint import pprint
from parallel_utils import map_layers_to_multi_gpus, get_lowest_occupied_gpu
import torch.nn as nn
from quantize.omniquant import omniquant
from tqdm import tqdm
import utils
from pathlib import Path
from categories import subcategories, categories

from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_opt_layer import QuantOPTDecoderLayer
from quantize.int_linear import QuantLinear

from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

torch.backends.cudnn.benchmark = True

net_choices = [
    "opt-125m",
    "opt-1.3b",
    "opt-2.7b",
    "opt-6.7b",
    "opt-13b",
    "opt-30b",
    "opt-66b",

    "llama-7b",
    "llama-13b",
    "llama-30b",
    "llama-65b",
    "Llama-2-7b",
    "Llama-2-13b",
    "Llama-2-70b",
    "Llama-2-7b-chat",
    "Llama-2-13b-chat",
    "llava-llama-2-13b-chat-lightning-preview",

    "falcon-180b",
    "falcon-7b",

    "mixtral-8x7b",

    "qwen3-8b",
    "qwen3-14b",
    "qwen3-32b",
]



# -----------------------------------------------------------------------------
# Inlined evaluation logic (ptq_evaluate-style blocks), no tools/eval.py needed
# -----------------------------------------------------------------------------
class _TokenBlocks(Dataset):
    def __init__(self, ids_1d: torch.Tensor, block_size: int, max_fraction: float = 1.0):
        assert ids_1d.ndim == 1
        self.ids = ids_1d
        self.block_size = int(block_size)

        nblocks_full = (len(self.ids) - 1) // self.block_size
        nblocks_full = max(int(nblocks_full), 0)

        if not (0.0 < float(max_fraction) <= 1.0):
            raise ValueError(f"max_fraction must be in (0,1], got {max_fraction}")

        self.nblocks = int(nblocks_full * float(max_fraction))
        self.nblocks = max(self.nblocks, 0)

    def __len__(self):
        return self.nblocks

    def __getitem__(self, idx: int):
        s = idx * self.block_size
        w = self.ids[s : s + self.block_size + 1]
        x = w[:-1].clone()
        y = w[1:].clone()
        return x, y


def _cache_path_for_tokens(cache_dir: str, model_family: str, ds_name: str, ds_subset: str, split: str):
    safe = f"{ds_name}_{ds_subset}_{split}".replace("/", "_").replace(":", "_")
    return os.path.join(cache_dir, f"eval_tokens_{model_family}_{safe}.pt")


def _load_eval_tokens(
    tokenizer,
    cache_dir: str,
    model_family: str,
    ds_name: str,
    ds_subset: str,
    split: str,
    text_field: str = "text",
):
    os.makedirs(cache_dir, exist_ok=True)
    cpath = _cache_path_for_tokens(cache_dir, model_family, ds_name, ds_subset, split)
    if os.path.exists(cpath):
        ids_1d = torch.load(cpath, map_location="cpu")
        if not isinstance(ids_1d, torch.Tensor):
            raise RuntimeError(f"Corrupt cache (not a tensor): {cpath}")
        return ids_1d

    ds = load_dataset(ds_name, ds_subset, split=split)
    if text_field not in ds.column_names:
        raise RuntimeError(
            f"Expected text field '{text_field}' in {ds_name}:{ds_subset}:{split}, got {ds.column_names}"
        )

    text = "\n\n".join(ds[text_field])
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    ids = enc.input_ids
    if ids.ndim != 2 or ids.shape[0] != 1:
        raise RuntimeError(f"Unexpected tokenized shape: {tuple(ids.shape)}")
    ids_1d = ids[0].contiguous().cpu()
    torch.save(ids_1d, cpath)
    return ids_1d


@torch.no_grad()
def _perplexity_over_loader(model, loader: DataLoader, device: str):
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    loss_fct = nn.CrossEntropyLoss(reduction="sum")

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        out = model(input_ids=x)
        logits = getattr(out, "logits", out[0])
        nll = loss_fct(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total_nll += float(nll.item())
        total_tokens += int(y.numel())

    ppl = float(np.exp(total_nll / max(total_tokens, 1)))
    return ppl, total_nll, total_tokens


@torch.no_grad()
def _eval_ppl_two_datasets(lm, args, logger):
    eval_specs = [
        ("wikitext", "wikitext-103-raw-v1", "test", "text"),
        ("lambada", "plain_text", "validation", "text"),
    ]
    block_size = 256
    batch_size_eval = 1
    eval_fraction = 1.0

    device = str(lm.device)

    use_cache = getattr(lm.model.config, "use_cache", None)
    if use_cache is not None:
        lm.model.config.use_cache = False

    results = {}
    for ds_name, ds_subset, split, text_field in eval_specs:
        key = f"{ds_name}:{ds_subset}::{split}"
        ids_1d = _load_eval_tokens(
            lm.tokenizer,
            cache_dir=args.cache_dir,
            model_family=args.model_family,
            ds_name=ds_name,
            ds_subset=ds_subset,
            split=split,
            text_field=text_field,
        )
        blocks = _TokenBlocks(ids_1d, block_size=block_size, max_fraction=eval_fraction)
        loader = DataLoader(blocks, batch_size=batch_size_eval, shuffle=False, drop_last=False)

        if len(blocks) == 0:
            logger.info(f"[PPL] {key}: not enough tokens for a single block (block_size={block_size})")
            results[key] = {"ppl": float("nan"), "nll": float("nan"), "tokens": 0, "blocks": 0}
            continue

        ppl, nll, ntok = _perplexity_over_loader(lm.model, loader, device=device)
        logger.info(f"[PPL] {key}  blocks={len(blocks)} tokens={ntok}  NLL={nll:.6f}  PPL={ppl:.4f}")
        results[key] = {"ppl": float(ppl), "nll": float(nll), "tokens": int(ntok), "blocks": int(len(blocks))}

    if use_cache is not None:
        lm.model.config.use_cache = use_cache

    return results


# -----------------------------------------------------------------------------
# Calibration dataset: wikitext-103-raw-v1 (train) in OmniQuant-style loader
# -----------------------------------------------------------------------------
def _cache_path_for_calib(cache_dir: str, model_family: str, model_id: str, seqlen: int, nsamples: int, seed: int):
    safe_model = model_id.replace("/", "_")
    return os.path.join(
        cache_dir,
        f"dataloader_{model_family}_wikitext103_train_seqlen{seqlen}_ns{nsamples}_seed{seed}_{safe_model}.pt",
    )


@torch.no_grad()
def _build_wikitext103_calib_loader(model_id: str, seqlen: int, nsamples: int, seed: int):
    traindata = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

    random.seed(seed)

    total_needed = int(nsamples) * int(seqlen)
    if total_needed <= 0:
        return []

    indices = list(range(len(traindata)))
    random.shuffle(indices)

    chunks = []
    n_tok = 0

    for idx in indices:
        txt = traindata[idx].get("text", "")
        if not isinstance(txt, str) or len(txt.strip()) == 0:
            continue

        enc = tokenizer(txt, return_tensors="pt", add_special_tokens=False)
        ids = enc.input_ids
        if ids.ndim != 2 or ids.shape[0] != 1:
            continue
        ids_1d = ids[0]
        if ids_1d.numel() == 0:
            continue

        chunks.append(ids_1d.cpu())
        n_tok += int(ids_1d.numel())

        if n_tok >= total_needed:
            break

    if n_tok < total_needed:
        raise RuntimeError(f"Not enough tokens collected: needed={total_needed}, got={n_tok}")

    stream = torch.cat(chunks, dim=0)[:total_needed].contiguous()

    trainloader = []
    for s in range(nsamples):
        start = s * seqlen
        end = start + seqlen
        inp = stream[start:end].unsqueeze(0)
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader


# -----------------------------------------------------------------------------
# Evaluation wrapper
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(lm, args, logger):
    results = {}

    if args.multigpu:
        if "opt" in args.net.lower():
            map_layers_to_multi_gpus(lm.model.model.decoder.layers)
            input_device = lm.model.model.decoder.layers[0].device
            output_device = lm.model.model.decoder.layers[-1].device
            lm._device = input_device
            assert input_device == output_device
            lm.model.model.decoder.embed_positions.to(input_device)
            lm.model.model.decoder.embed_tokens.to(input_device)
            lm.model.model.decoder.final_layer_norm.to(output_device)
            lm.model.lm_head.to(output_device)

        elif "llama" in args.net.lower() or "mixtral" in args.net.lower():
            map_layers_to_multi_gpus(lm.model.model.layers)
            input_device = lm.model.model.layers[0].device
            output_device = lm.model.model.layers[-1].device
            assert input_device == output_device
            lm._device = input_device
            lm.model.model.embed_tokens.to(input_device)
            lm.model.model.norm.to(output_device)
            lm.model.lm_head.to(output_device)

        elif "falcon" in args.net.lower():
            map_layers_to_multi_gpus(lm.model.transformer.h)
            input_device = lm.model.transformer.h[0].device
            output_device = lm.model.transformer.h[-1].device
            assert input_device == output_device
            lm._device = input_device
            lm.model.transformer.word_embeddings.to(input_device)
            lm.model.transformer.ln_f.to(output_device)
            lm.model.lm_head.to(output_device)
    else:
        if "opt" in args.net.lower():
            lm.model.model.decoder = lm.model.model.decoder.to(lm.device)
        elif "llama" in args.net.lower() or "mixtral" in args.net.lower():
            lm.model = lm.model.to(lm.device)
        elif "falcon" in args.net.lower():
            lm.model.transformer = lm.model.transformer.to(lm.device)

    if args.eval_ppl:
        results["ppl"] = _eval_ppl_two_datasets(lm, args, logger)

    if args.tasks != "":
        t_results = evaluator.simple_evaluate(
            lm,
            tasks=args.tasks,
            num_fewshot=args.num_fewshot,
            limit=None if args.limit == -1 else args.limit,
        )
        results.update(t_results)
        logger.info(results)
        pprint(results)

    return results


@torch.no_grad()
def evaluate_ppl_only(lm, args, logger, tag: str):
    class _ArgsView:
        pass

    a = _ArgsView()
    for k, v in vars(args).items():
        setattr(a, k, v)
    a.eval_ppl = True
    a.tasks = ""
    logger.info(f"=== PPL evaluation ({tag}) ===")
    return evaluate(lm, a, logger)


def _parse_float_grid(s: str):
    # "1e-4,1e-3,1e-2"
    if s is None or s.strip() == "":
        return None
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok == "":
            continue
        out.append(float(tok))
    return out


def _cleanup_logger(logger):
    # utils.create_logger in many repos attaches handlers; remove them between sweep runs
    try:
        handlers = list(logger.handlers)
        for h in handlers:
            try:
                h.flush()
                h.close()
            except Exception:
                pass
            try:
                logger.removeHandler(h)
            except Exception:
                pass
    except Exception:
        pass


def _run_one_setting(base_args, let_lr: float, lwc_lr: float):
    # Create per-run args + dirs
    args = copy.deepcopy(base_args)

    # override hyperparams
    args.let_lr = float(let_lr)
    args.lwc_lr = float(lwc_lr)

    # unique run folder
    run_name = f"letlr_{let_lr:.0e}_lwclr_{lwc_lr:.0e}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # write save_dir under run_dir if requested
    if args.save_dir is not None:
        args.save_dir = str(run_dir / "hf_ckpt")

    # new logger per run
    logger = utils.create_logger(run_dir)
    logger.info(f"=== SWEEP RUN: {run_name} ===")
    logger.info(args)

    # determinism
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # load model fresh each run
    if args.net is None:
        args.net = args.model.split("/")[-1]
    args.model_family = args.net.split("-")[0]
    lm = LMClass(args)
    lm.seqlen = 2048
    lm.model.eval()

    # keep your current behavior (you had this in the script)
    lm.model = lm.model.float()

    for param in lm.model.parameters():
        param.requires_grad = False

    # (re)build quant params dicts
    args.weight_quant_params = {
        "n_bits": args.wbits,
        "per_channel_axes": [0],
        "symmetric": args.symmetric,
        "dynamic_method": args.w_dynamic_method,
        "group_size": args.group_size,
        "lwc": args.lwc,
        "disable_zero_point": args.disable_zero_point,
    }
    args.act_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": True,
        "dynamic_method": args.a_dynamic_method,
        "disable_zero_point": args.disable_zero_point,
    }
    args.q_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": True,
        "dynamic_method": args.a_dynamic_method,
    }
    args.k_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": True,
        "dynamic_method": args.a_dynamic_method,
    }
    args.v_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": True,
        "dynamic_method": args.a_dynamic_method,
    }
    args.p_quant_params = {
        "n_bits": 16,
        "metric": "fix0to1",
    }

    if args.multigpu:
        gpu_id = get_lowest_occupied_gpu(wait_memory=5000)
        lm._device = f"cuda:{gpu_id}"
        logger.info(f"set quantization in gpu {gpu_id}")

    # act scales / shifts
    if args.act_scales is None:
        args.act_scales = f"./act_scales/{args.net}.pt"
    if args.act_shifts is None:
        args.act_shifts = f"./act_shifts/{args.net}.pt"

    # calibration loader
    if args.calib_dataset == "wikitext103":
        cpath = _cache_path_for_calib(args.cache_dir, args.model_family, args.model, lm.seqlen, args.nsamples, args.seed)
        if os.path.exists(cpath):
            dataloader = torch.load(cpath, map_location="cpu")
            logger.info(f"load calibration from {cpath}")
        else:
            dataloader = _build_wikitext103_calib_loader(args.model, lm.seqlen, args.nsamples, args.seed)
            torch.save(dataloader, cpath)
            logger.info(f"saved calibration to {cpath}")
    else:
        cache_dataloader = f"{args.cache_dir}/dataloader_{args.model_family}_{args.calib_dataset}_{args.nsamples}.cache"
        if os.path.exists(cache_dataloader):
            dataloader = torch.load(cache_dataloader)
            logger.info(f"load calibration from {cache_dataloader}")
        else:
            dataloader, _ = get_loaders(
                args.calib_dataset,
                nsamples=args.nsamples,
                seed=args.seed,
                model=args.model,
                seqlen=lm.seqlen,
            )
            torch.save(dataloader, cache_dataloader)

    act_scales = None
    act_shifts = None
    if args.let:
        act_scales = torch.load(args.act_scales)
        act_shifts = torch.load(args.act_shifts)

    tick = time.time()
    logger.info("=== start quantization ===")
    omniquant(lm, args, dataloader, act_scales, act_shifts, logger)
    logger.info(f"quantization done in {time.time() - tick:.2f}s")

    # save ckpt if requested
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        lm.model.save_pretrained(args.save_dir)
        lm.tokenizer.save_pretrained(args.save_dir)

    # eval ppl
    ppl_results = evaluate_ppl_only(lm, args, logger, tag="quantized_post_finetune")

    # write per-run results
    out_json = {
        "let_lr": let_lr,
        "lwc_lr": lwc_lr,
        "run_dir": str(run_dir),
        "ppl": ppl_results.get("ppl", {}),
    }
    with open(run_dir / "results.json", "w") as f:
        json.dump(out_json, f, indent=2)

    # cleanup
    _cleanup_logger(logger)
    del lm
    torch.cuda.empty_cache()
    gc.collect()

    return out_json


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model name of model path")
    parser.add_argument("--cache_dir", default="./cache", type=str)
    parser.add_argument("--output_dir", default="../log/", type=str)
    parser.add_argument("--save_dir", default=None, type=str)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--real_quant", default=False, action="store_true")

    parser.add_argument(
        "--calib_dataset",
        type=str,
        default="wikitext103",
        choices=["wikitext103", "wikitext2", "ptb", "c4", "mix", "pile"],
    )
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--tasks", default="")
    parser.add_argument("--eval_ppl", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--wbits", type=int, default=4)
    parser.add_argument("--abits", type=int, default=16)
    parser.add_argument("--group_size", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--let_lr", type=float, default=5e-3)
    parser.add_argument("--lwc_lr", type=float, default=1e-2)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--let", default=False, action="store_true")
    parser.add_argument("--lwc", default=False, action="store_true")
    parser.add_argument("--aug_loss", default=False, action="store_true")
    parser.add_argument("--symmetric", default=False, action="store_true")
    parser.add_argument("--disable_zero_point", default=False, action="store_true")
    parser.add_argument("--a_dynamic_method", type=str, default="per_token", choices=["per_token"])
    parser.add_argument("--w_dynamic_method", type=str, default="per_channel", choices=["per_channel"])
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--multigpu", action="store_true")
    parser.add_argument("--deactive_amp", action="store_true")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        required=False,
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
    )
    parser.add_argument("--net", type=str, default=None, choices=net_choices)
    parser.add_argument("--act-scales", type=str, default=None)
    parser.add_argument("--act-shifts", type=str, default=None)
    parser.add_argument("--eval_quant_before_ft", action="store_true")

    parser.add_argument("--let_lr_grid", type=str, default="1e-4,1e-3,1e-2")
    parser.add_argument("--lwc_lr_grid", type=str, default="1e-4,1e-3,1e-2")

    args = parser.parse_args()

    # deactive_amp logic kept from your script
    if (args.wbits < 16 and args.wbits >= 8) or (args.abits < 16 and args.abits >= 8):
        args.deactive_amp = True

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # parse grids
    let_grid = _parse_float_grid(args.let_lr_grid) or [1e-4, 1e-3, 1e-2]
    lwc_grid = _parse_float_grid(args.lwc_lr_grid) or [1e-4, 1e-3, 1e-2]

    # run sweep
    sweep_results = []
    for let_lr in let_grid:
        for lwc_lr in lwc_grid:
            sweep_results.append(_run_one_setting(args, let_lr=let_lr, lwc_lr=lwc_lr))

    # summarize: sort by wikitext ppl (if available)
    def get_wt_ppl(r):
        try:
            return r["ppl"]["wikitext:wikitext-103-raw-v1::test"]["ppl"]
        except Exception:
            return float("inf")

    sweep_results_sorted = sorted(sweep_results, key=get_wt_ppl)

    summary_path = Path(args.output_dir) / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "let_lr_grid": let_grid,
                "lwc_lr_grid": lwc_grid,
                "results": sweep_results_sorted,
            },
            f,
            indent=2,
        )

    # Print a compact "range" view
    wt_ppls = [get_wt_ppl(r) for r in sweep_results if np.isfinite(get_wt_ppl(r))]
    lb_ppls = []
    for r in sweep_results:
        try:
            lb_ppls.append(r["ppl"]["lambada:plain_text::validation"]["ppl"])
        except Exception:
            pass
    lb_ppls = [x for x in lb_ppls if np.isfinite(x)]

    print("============================================================")
    print(f"Sweep done. Wrote: {summary_path}")
    if wt_ppls:
        print(f"Wikitext PPL range: min={min(wt_ppls):.4f}  max={max(wt_ppls):.4f}")
    if lb_ppls:
        print(f"Lambada  PPL range: min={min(lb_ppls):.4f}  max={max(lb_ppls):.4f}")
    print("Top-5 by Wikitext PPL:")
    for r in sweep_results_sorted[:5]:
        wt = get_wt_ppl(r)
        try:
            lb = r["ppl"]["lambada:plain_text::validation"]["ppl"]
        except Exception:
            lb = float("nan")
        print(f"  let_lr={r['let_lr']:.0e} lwc_lr={r['lwc_lr']:.0e}  WT={wt:.4f}  LB={lb:.4f}  dir={r['run_dir']}")
    print("============================================================")


if __name__ == "__main__":
    print(sys.argv)
    main()
