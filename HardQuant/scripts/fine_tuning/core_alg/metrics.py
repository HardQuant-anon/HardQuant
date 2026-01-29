#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# HF datasets (used by tools/eval.py loaders)
from datasets import load_dataset  # type: ignore


# -----------------------------------------------------------------------------
# Repo path setup (matches your training script)
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[2]
TOOLS_DIR = PROJECT_DIR / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from eval import (  # type: ignore
    build_eval_loader,
    build_train_loader,
    perplexity,
)
from fake_quant import (  # type: ignore
    quantize_model as fake_quantize_model,
    get_output_embedding_module,
)

# Robust imports across transformers versions (for family checks / optional logic)
try:
    from transformers.models.opt.modeling_opt import OPTPreTrainedModel  # type: ignore
except Exception:
    OPTPreTrainedModel = None  # type: ignore[assignment]

try:
    from transformers.models.llama.modeling_llama import LlamaPreTrainedModel  # type: ignore
except Exception:
    LlamaPreTrainedModel = None  # type: ignore[assignment]

try:
    from transformers.models.mistral.modeling_mistral import MistralPreTrainedModel  # type: ignore
except Exception:
    MistralPreTrainedModel = None  # type: ignore[assignment]

try:
    from transformers.models.mixtral.modeling_mixtral import MixtralPreTrainedModel  # type: ignore
except Exception:
    MixtralPreTrainedModel = None  # type: ignore[assignment]

# Qwen support (best-effort; falls back to model_type check)
try:
    from transformers.models.qwen2.modeling_qwen2 import Qwen2PreTrainedModel  # type: ignore
except Exception:
    Qwen2PreTrainedModel = None  # type: ignore[assignment]

try:
    from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoePreTrainedModel  # type: ignore
except Exception:
    Qwen2MoePreTrainedModel = None  # type: ignore[assignment]


DEFAULTS = {
    "model": "facebook/opt-1.3b",
    "dataset_in": "wikitext:wikitext-103-raw-v1",
    "dataset_in_train_split": "train",
    "dataset_in_eval_split": "test",
    "block_size": 1024,
    "batch_size_eval": 1,
    "batch_size_calib": 1,
    "calib_tokens": 500_000,
    "eval_fraction": 0.01,
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    # quant config (for fake-quant model forward)
    "weight_quant": "per_channel",
    "act_quant": "per_token",
    "quantize_bmm_input": False,
    "weight_bits": 8,
    "act_bits": 8,
    "eps": 1e-12,
    # hardness smooth option
    "rx_use_smooth": False,
    "rx_smooth_p": 8.0,
    # how many batches to use for layerwise metrics
    "layerwise_batches": 8,
    # saving
    "output_dir": None,
    # whether to compute baseline PPLs
    "compute_ppl": True,
    # NEW: dtype controls GPU memory
    "dtype": "bf16",  # bf16 is safest on H100/H200/A100; use fp16 if no bf16
    # NEW: avoid writing huge arrays into results.json
    "embed_distributions_in_results": False,
}


# -----------------------------------------------------------------------------
# Small utils
# -----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cycle(loader: Iterable):
    while True:
        for batch in loader:
            yield batch


def _release_cuda_models(*models) -> None:
    for m in models:
        if m is not None:
            del m
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _validate_fraction(frac: float) -> float:
    f = float(frac)
    if not (0.0 < f <= 1.0):
        raise ValueError(f"eval_fraction must be in (0,1], got {frac}")
    return f


def parse_dataset(arg: str) -> Tuple[str, Optional[str]]:
    if ":" in arg:
        name, subset = arg.split(":", 1)
        return name, subset or None
    return arg, None


def hf_tuple(dataset: Tuple[str, Optional[str]]) -> Tuple[str, ...]:
    return (dataset[0], dataset[1]) if dataset[1] is not None else (dataset[0],)


def _make_tokenizer(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    return tok


def _model_type(model: nn.Module) -> Optional[str]:
    return getattr(getattr(model, "config", None), "model_type", None)


def _is_supported_causallm(model: nn.Module) -> bool:
    if OPTPreTrainedModel is not None and isinstance(model, OPTPreTrainedModel):
        return True
    if LlamaPreTrainedModel is not None and isinstance(model, LlamaPreTrainedModel):
        return True
    if MistralPreTrainedModel is not None and isinstance(model, MistralPreTrainedModel):
        return True
    if MixtralPreTrainedModel is not None and isinstance(model, MixtralPreTrainedModel):
        return True
    if Qwen2PreTrainedModel is not None and isinstance(model, Qwen2PreTrainedModel):
        return True
    if Qwen2MoePreTrainedModel is not None and isinstance(model, Qwen2MoePreTrainedModel):
        return True
    mt = (_model_type(model) or "").lower()
    return mt in {"opt", "llama", "mistral", "mixtral", "qwen2", "qwen2_moe", "qwen", "qwen3"}


def _resolve_dtype(s: str) -> torch.dtype:
    s = (s or "").lower()
    if s in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if s in {"fp16", "float16", "half"}:
        return torch.float16
    if s in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unknown dtype {s}. Use bf16|fp16|fp32.")


class DisableDropout:
    _DROPOUT_TYPES = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)

    def __init__(self, model: nn.Module):
        self.model = model
        self._states: List[Tuple[nn.Module, bool]] = []

    def __enter__(self):
        self._states.clear()
        for m in self.model.modules():
            if isinstance(m, self._DROPOUT_TYPES):
                self._states.append((m, m.training))
                m.eval()
        return self

    def __exit__(self, exc_type, exc, tb):
        for m, was_training in self._states:
            m.train() if was_training else m.eval()
        self._states.clear()
        return False


# -----------------------------------------------------------------------------
# Decoder layer access
# -----------------------------------------------------------------------------
def _unwrap_wrapped_model(model: nn.Module) -> nn.Module:
    cur: nn.Module = model
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        while isinstance(cur, FSDP):
            cur = cur.module  # type: ignore[assignment]
    except Exception:
        pass
    if isinstance(cur, torch.nn.parallel.DistributedDataParallel):
        cur = cur.module
    return cur


def _get_attr_path(root: nn.Module, path: List[str]) -> Optional[object]:
    cur: object = root
    for a in path:
        cur = getattr(cur, a, None)
        if cur is None:
            return None
    return cur


def _is_opt_family(model: nn.Module) -> bool:
    return (_model_type(model) or "").lower() == "opt"


def _get_decoder_layers(model: nn.Module):
    model = _unwrap_wrapped_model(model)

    opt_paths = [
        ["model", "decoder", "layers"],
        ["model", "model", "decoder", "layers"],
        ["base_model", "model", "decoder", "layers"],
        ["base_model", "model", "model", "decoder", "layers"],
    ]

    llama_like_paths = [
        ["model", "layers"],
        ["model", "model", "layers"],
        ["base_model", "model", "layers"],
        ["base_model", "model", "model", "layers"],
    ]

    qwen_fallback_paths = [
        ["transformer", "h"],
        ["model", "transformer", "h"],
        ["base_model", "transformer", "h"],
        ["base_model", "model", "transformer", "h"],
    ]

    paths = (opt_paths + llama_like_paths) if _is_opt_family(model) else (llama_like_paths + opt_paths)
    paths = paths + qwen_fallback_paths

    for p in paths:
        cur = _get_attr_path(model, p)
        if isinstance(cur, (list, nn.ModuleList)):
            return cur
    raise ValueError("Could not locate decoder layers.")


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def _rx_rowwise_values(x: torch.Tensor, eps: float, use_smooth: bool, smooth_p: float) -> torch.Tensor:
    if x.numel() == 0:
        return x.new_zeros((0,), dtype=torch.float32)

    # NOTE: we avoid an explicit clone; we only cast to fp32 for the math.
    xf = x.to(dtype=torch.float32)
    xr = xf.view(1, -1) if xf.dim() == 1 else xf.view(-1, xf.size(-1))
    c = int(xr.size(-1))
    if c <= 0:
        return x.new_zeros((0,), dtype=torch.float32)

    abs_x = xr.abs()
    if use_smooth:
        maxv = abs_x.pow(smooth_p).sum(dim=-1).add(eps).pow(1.0 / smooth_p)
    else:
        maxv = abs_x.amax(dim=-1)

    l2 = (xr * xr).sum(dim=-1).add(eps).sqrt()
    denom = l2 / math.sqrt(float(c))
    ratios = (maxv / denom).pow(2)
    ratios = torch.nan_to_num(ratios, nan=0.0, posinf=0.0, neginf=0.0).clamp_(0.0, 1e12)
    return ratios.to(dtype=torch.float32)


def _rx_rowwise_sum_count(x: torch.Tensor, eps: float, use_smooth: bool, smooth_p: float) -> Tuple[torch.Tensor, int]:
    r = _rx_rowwise_values(x, eps=eps, use_smooth=use_smooth, smooth_p=smooth_p)
    if r.numel() == 0:
        return x.new_zeros(()), 0
    return r.sum(dtype=torch.float32), int(r.numel())


def _fake_quantize_activation_fp32(x: torch.Tensor, bits: int, act_quant: str, eps: float) -> torch.Tensor:
    if bits is None or int(bits) <= 0:
        return x
    b = int(bits)
    qmax = (1 << (b - 1)) - 1
    if qmax <= 0:
        return x

    xf = x.to(dtype=torch.float32)

    if act_quant == "per_tensor":
        amax = xf.abs().amax()
        clip = amax.clamp(min=eps)
        scale = clip / float(qmax)
        q = (xf / scale).round().clamp(-qmax, qmax) * scale
        return q

    if act_quant != "per_token":
        raise ValueError(f"act_quant must be per_token|per_tensor, got {act_quant}")

    xr = xf.view(1, -1) if xf.dim() == 1 else xf.view(-1, xf.size(-1))
    amax = xr.abs().amax(dim=-1, keepdim=True)
    clip = amax.clamp(min=eps)
    scale = clip / float(qmax)
    q = (xr / scale).round().clamp(-qmax, qmax) * scale
    return q.view_as(xf)


def _xq_rowwise_sum_count(x: torch.Tensor, bits: int, act_quant: str, eps: float) -> Tuple[torch.Tensor, int]:
    if x.numel() == 0:
        return x.new_zeros(()), 0

    xf = x.to(dtype=torch.float32)
    xr = xf.view(1, -1) if xf.dim() == 1 else xf.view(-1, xf.size(-1))

    q = _fake_quantize_activation_fp32(xr, bits=bits, act_quant=act_quant, eps=eps)
    diff = xr - q

    num = diff.pow(2).sum(dim=-1).add(eps).sqrt()
    den = xr.pow(2).sum(dim=-1).add(eps).sqrt()
    ratio = (num / den)
    ratio = torch.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0).clamp_(0.0, 1e12)
    return ratio.sum(dtype=torch.float32), int(ratio.numel())


# -----------------------------------------------------------------------------
# Collectors
# -----------------------------------------------------------------------------
class BlockInputTokenRxCollector:
    def __init__(self, model: nn.Module, eps: float, use_smooth: bool, smooth_p: float):
        self.eps = float(eps)
        self.use_smooth = bool(use_smooth)
        self.smooth_p = float(smooth_p)
        self._rx_vals_cpu: Dict[int, List[torch.Tensor]] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

        layers = _get_decoder_layers(model)
        for j, block in enumerate(layers):

            def make_prehook(layer_idx: int):
                def prehook(_module: nn.Module, inp):
                    if not isinstance(inp, (tuple, list)) or len(inp) == 0:
                        return
                    x = inp[0]
                    if not isinstance(x, torch.Tensor):
                        return
                    r = _rx_rowwise_values(x.detach(), eps=self.eps, use_smooth=self.use_smooth, smooth_p=self.smooth_p)
                    if r.numel() == 0:
                        return
                    self._rx_vals_cpu.setdefault(layer_idx, []).append(r.detach().cpu())
                return prehook

            self._handles.append(block.register_forward_pre_hook(make_prehook(j)))

    def result(self) -> Dict[int, List[float]]:
        out: Dict[int, List[float]] = {}
        for k in sorted(self._rx_vals_cpu.keys()):
            cat = torch.cat(self._rx_vals_cpu[k], dim=0) if len(self._rx_vals_cpu[k]) > 1 else self._rx_vals_cpu[k][0]
            out[k] = cat.to(dtype=torch.float32).tolist()
        return out

    def remove(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()
        self._rx_vals_cpu.clear()


@torch.no_grad()
def collect_block_input_rx_tokens(
    model: nn.Module,
    batches_cpu: List[torch.LongTensor],
    device: str,
    eps: float,
    use_smooth: bool,
    smooth_p: float,
) -> Dict[int, List[float]]:
    was_training = model.training
    model.eval()

    collector = BlockInputTokenRxCollector(model, eps=eps, use_smooth=use_smooth, smooth_p=smooth_p)
    try:
        with DisableDropout(model):
            for x_cpu in batches_cpu:
                x = x_cpu.to(device, non_blocking=True)
                _ = model(input_ids=x, use_cache=False)
    finally:
        out = collector.result()
        collector.remove()
        model.train(was_training)

    return out


def _is_weight_module_for_rw(m: nn.Module) -> bool:
    if isinstance(m, nn.Linear):
        return True
    if hasattr(m, "weight") and isinstance(getattr(m, "weight", None), torch.Tensor):
        w = getattr(m, "weight")
        if isinstance(w, torch.Tensor) and w.dim() == 2:
            if isinstance(m, nn.Embedding):
                return False
            return True
    return False


@torch.no_grad()
def collect_rw_over_output_channels(
    model: nn.Module,
    eps: float,
    use_smooth: bool,
    smooth_p: float,
    exclude_output_embedding: bool = True,
) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}

    out_emb = None
    if exclude_output_embedding:
        try:
            out_emb = get_output_embedding_module(model)
        except Exception:
            out_emb = None

    for name, module in model.named_modules():
        if not _is_weight_module_for_rw(module):
            continue
        if exclude_output_embedding and out_emb is not None and module is out_emb:
            continue

        w = getattr(module, "weight", None)
        if not isinstance(w, torch.Tensor) or w.dim() != 2:
            continue

        r = _rx_rowwise_values(w.detach(), eps=float(eps), use_smooth=bool(use_smooth), smooth_p=float(smooth_p))
        if r.numel() == 0:
            continue
        out[name] = r.detach().cpu().tolist()

    return out


class BlockInputCollector:
    def __init__(
        self,
        model: nn.Module,
        eps: float,
        use_smooth: bool,
        smooth_p: float,
        collect_xq: bool,
        act_bits: int,
        act_quant: str,
    ):
        self.eps = float(eps)
        self.use_smooth = bool(use_smooth)
        self.smooth_p = float(smooth_p)
        self.collect_xq = bool(collect_xq)
        self.act_bits = int(act_bits)
        self.act_quant = str(act_quant)

        self._rx_sum: Dict[int, torch.Tensor] = {}
        self._rx_cnt: Dict[int, int] = {}
        self._xq_sum: Dict[int, torch.Tensor] = {}
        self._xq_cnt: Dict[int, int] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

        layers = _get_decoder_layers(model)
        for j, block in enumerate(layers):

            def make_prehook(layer_idx: int):
                def prehook(_module: nn.Module, inp):
                    if not isinstance(inp, (tuple, list)) or len(inp) == 0:
                        return
                    x = inp[0]
                    if not isinstance(x, torch.Tensor):
                        return

                    # NOTE: avoid .clone() (unnecessary for pure read)
                    x0 = x.detach()

                    rx_s, rx_c = _rx_rowwise_sum_count(x0, eps=self.eps, use_smooth=self.use_smooth, smooth_p=self.smooth_p)
                    if rx_c > 0:
                        self._rx_sum[layer_idx] = self._rx_sum.get(layer_idx, rx_s.detach()) + rx_s.detach()
                        self._rx_cnt[layer_idx] = self._rx_cnt.get(layer_idx, 0) + rx_c

                    if self.collect_xq:
                        xq_s, xq_c = _xq_rowwise_sum_count(x0, bits=self.act_bits, act_quant=self.act_quant, eps=self.eps)
                        if xq_c > 0:
                            self._xq_sum[layer_idx] = self._xq_sum.get(layer_idx, xq_s.detach()) + xq_s.detach()
                            self._xq_cnt[layer_idx] = self._xq_cnt.get(layer_idx, 0) + xq_c

                return prehook

            self._handles.append(block.register_forward_pre_hook(make_prehook(j)))

    def result_rx(self) -> Dict[int, float]:
        return {k: float((self._rx_sum[k] / float(self._rx_cnt[k])).detach().cpu()) for k in sorted(self._rx_cnt.keys())}

    def result_xq(self) -> Dict[int, float]:
        return {k: float((self._xq_sum[k] / float(self._xq_cnt[k])).detach().cpu()) for k in sorted(self._xq_cnt.keys())}

    def remove(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()


@torch.no_grad()
def collect_block_input_metrics(
    model: nn.Module,
    batches_cpu: List[torch.LongTensor],
    device: str,
    eps: float,
    use_smooth: bool,
    smooth_p: float,
    collect_xq: bool,
    act_bits: int,
    act_quant: str,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    was_training = model.training
    model.eval()

    collector = BlockInputCollector(
        model,
        eps=eps,
        use_smooth=use_smooth,
        smooth_p=smooth_p,
        collect_xq=collect_xq,
        act_bits=act_bits,
        act_quant=act_quant,
    )

    try:
        with DisableDropout(model):
            for x_cpu in batches_cpu:
                x = x_cpu.to(device, non_blocking=True)
                _ = model(input_ids=x, use_cache=False)
    finally:
        rx = collector.result_rx()
        xq = collector.result_xq()
        collector.remove()
        model.train(was_training)

    return rx, xq


def sample_calib_batches_cpu(iterator: Iterable, n_batches: int) -> List[torch.LongTensor]:
    out: List[torch.LongTensor] = []
    for _ in range(int(n_batches)):
        (x,) = next(iterator)
        out.append(x.detach().to("cpu"))
    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Collect layerwise R(X), R(prefix-quant X), and ||X-Q(X)||/||X|| at block inputs, "
            "plus distributions of per-token R(X) and per-output-channel R(W)."
        )
    )

    p.add_argument("--model", type=str, default=DEFAULTS["model"])
    p.add_argument("--dataset-in", type=str, default=DEFAULTS["dataset_in"])
    p.add_argument("--dataset-in-train-split", type=str, default=DEFAULTS["dataset_in_train_split"])
    p.add_argument("--dataset-in-eval-split", type=str, default=DEFAULTS["dataset_in_eval_split"])

    p.add_argument("--block-size", type=int, default=DEFAULTS["block_size"])
    p.add_argument("--batch-size-eval", type=int, default=DEFAULTS["batch_size_eval"])
    p.add_argument("--batch-size-calib", type=int, default=DEFAULTS["batch_size_calib"])
    p.add_argument("--calib-tokens", type=int, default=DEFAULTS["calib_tokens"])

    p.add_argument("--eval-fraction", type=float, default=DEFAULTS["eval_fraction"])
    p.add_argument("--device", type=str, default=DEFAULTS["device"])
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])

    p.add_argument("--weight-quant", type=str, default=DEFAULTS["weight_quant"], choices=["per_channel", "per_tensor"])
    p.add_argument("--act-quant", type=str, default=DEFAULTS["act_quant"], choices=["per_token", "per_tensor"])
    p.add_argument("--quantize-bmm-input", action="store_true", default=DEFAULTS["quantize_bmm_input"])
    p.add_argument("--weight-bits", type=int, default=DEFAULTS["weight_bits"])
    p.add_argument("--act-bits", type=int, default=DEFAULTS["act_bits"])
    p.add_argument("--eps", type=float, default=DEFAULTS["eps"])

    p.add_argument("--rx-use-smooth", action="store_true", default=DEFAULTS["rx_use_smooth"])
    p.add_argument("--rx-smooth-p", type=float, default=DEFAULTS["rx_smooth_p"])

    p.add_argument("--layerwise-batches", type=int, default=DEFAULTS["layerwise_batches"])

    p.add_argument(
        "--compute-ppl",
        action=argparse.BooleanOptionalAction,
        default=DEFAULTS["compute_ppl"],
        help="Compute PPLs. For Qwen-14B, disable to save memory/time.",
    )

    # NEW: dtype (big memory lever)
    p.add_argument("--dtype", type=str, default=DEFAULTS["dtype"], choices=["bf16", "fp16", "fp32"])

    # NEW: don’t embed big arrays inside results.json by default
    p.add_argument(
        "--embed-distributions-in-results",
        action=argparse.BooleanOptionalAction,
        default=DEFAULTS["embed_distributions_in_results"],
        help="If true, also embed rx_tokens/rw_channels dicts into results.json (can be huge).",
    )

    out_default = DEFAULTS["output_dir"] or (PROJECT_DIR / "outputs" / f"metrics_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    p.add_argument("--output-dir", type=Path, default=out_default)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.eval_fraction = _validate_fraction(args.eval_fraction)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    dtype = _resolve_dtype(args.dtype)

    dataset_in = parse_dataset(args.dataset_in)

    tok = _make_tokenizer(args.model)

    # NEW: load in bf16/fp16 to cut memory, and with low_cpu_mem_usage
    model_fp = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(args.device).eval()

    if not _is_supported_causallm(model_fp):
        raise ValueError(
            f"Expected OPT/LLaMA/Mistral/Mixtral/Qwen-family. "
            f"Got type={type(model_fp)} model_type={_model_type(model_fp)}"
        )

    # loaders
    train_loader = build_train_loader(
        tok,
        hf_tuple(dataset_in),
        args.block_size,
        args.calib_tokens,
        args.batch_size_calib,
        split=args.dataset_in_train_split,
    )
    train_iter = cycle(train_loader)

    eval_loader_in, nblocks_in, ntok_in = build_eval_loader(
        tok,
        hf_tuple(dataset_in),
        args.block_size,
        args.batch_size_eval,
        split=args.dataset_in_eval_split,
        max_fraction=args.eval_fraction,
    )

    results: Dict[str, Any] = {
        "model": args.model,
        "model_type": _model_type(model_fp),
        "dtype": str(dtype).replace("torch.", ""),
        "dataset_in": args.dataset_in,
        "dataset_in_train_split": args.dataset_in_train_split,
        "dataset_in_eval_split": args.dataset_in_eval_split,
        "block_size": args.block_size,
        "batch_size_eval": args.batch_size_eval,
        "batch_size_calib": args.batch_size_calib,
        "calib_tokens": args.calib_tokens,
        "eval_fraction": args.eval_fraction,
        "weight_quant": args.weight_quant,
        "act_quant": args.act_quant,
        "weight_bits": args.weight_bits,
        "act_bits": args.act_bits,
        "eps": args.eps,
        "rx_use_smooth": bool(args.rx_use_smooth),
        "rx_smooth_p": float(args.rx_smooth_p),
        "nblocks_in": nblocks_in,
        "tokens_in": ntok_in,
    }

    # Optional PPLs (NOTE: no deepcopy; we do dense first, then quantize in-place and do fakequant)
    if bool(args.compute_ppl):
        with torch.no_grad():
            ppl_fp32_in, nll_fp32_in, _ = perplexity(model_fp, eval_loader_in, args.device)
        results["ppl_fp32_in"] = float(ppl_fp32_in)
        results["nll_fp32_in"] = float(nll_fp32_in)
        print(f"[DENSE] in_nll={nll_fp32_in:.4f} in_ppl={ppl_fp32_in:.4f}")

    # Sample identical calibration batches once (CPU)
    print(f"[CALIB] Sampling {int(args.layerwise_batches)} batches (stored on CPU for consistent inputs)...")
    calib_batches_cpu = sample_calib_batches_cpu(train_iter, n_batches=int(args.layerwise_batches))

    # -------------------------
    # DENSE measurements FIRST
    # -------------------------
    print("[DENSE] collecting average R(X) and ||X-Q(X)||/||X|| at block inputs...")
    rx_dense, xq_dense = collect_block_input_metrics(
        model=model_fp,
        batches_cpu=calib_batches_cpu,
        device=args.device,
        eps=args.eps,
        use_smooth=bool(args.rx_use_smooth),
        smooth_p=float(args.rx_smooth_p),
        collect_xq=True,
        act_bits=int(args.act_bits),
        act_quant=str(args.act_quant),
    )

    print("[DENSE] collecting per-token R(X) at block inputs...")
    rx_tokens_dense = collect_block_input_rx_tokens(
        model=model_fp,
        batches_cpu=calib_batches_cpu,
        device=args.device,
        eps=args.eps,
        use_smooth=bool(args.rx_use_smooth),
        smooth_p=float(args.rx_smooth_p),
    )

    print("[DENSE] collecting per-output-channel R(W) over weight rows...")
    rw_channels_fp = collect_rw_over_output_channels(
        model=model_fp,
        eps=args.eps,
        use_smooth=bool(args.rx_use_smooth),
        smooth_p=float(args.rx_smooth_p),
        exclude_output_embedding=True,
    )

    # -----------------------------------------
    # QUANTIZE IN-PLACE (big memory saver)
    # -----------------------------------------
    print("[INPLACE-QUANT] quantizing model in-place (saves GPU memory vs deepcopy)...")
    model_fp = fake_quantize_model(
        model_fp,
        weight_quant=args.weight_quant,
        act_quant=args.act_quant,
        quantize_bmm_input=args.quantize_bmm_input,
        weight_bits=args.weight_bits,
        act_bits=args.act_bits,
        exclude_lm_head=True,
    ).eval()
    model_q = model_fp  # alias (same object)

    # Prefix-quant measurements on same in-place quantized model
    print("[PREFIX] collecting average R(X) at block inputs under quantized prefix...")
    rx_prefix, _ = collect_block_input_metrics(
        model=model_q,
        batches_cpu=calib_batches_cpu,
        device=args.device,
        eps=args.eps,
        use_smooth=bool(args.rx_use_smooth),
        smooth_p=float(args.rx_smooth_p),
        collect_xq=False,
        act_bits=int(args.act_bits),
        act_quant=str(args.act_quant),
    )

    print("[PREFIX] collecting per-token R(X) at block inputs...")
    rx_tokens_prefix = collect_block_input_rx_tokens(
        model=model_q,
        batches_cpu=calib_batches_cpu,
        device=args.device,
        eps=args.eps,
        use_smooth=bool(args.rx_use_smooth),
        smooth_p=float(args.rx_smooth_p),
    )

    print("[QMODEL] collecting per-output-channel R(W) over weight rows...")
    rw_channels_q = collect_rw_over_output_channels(
        model=model_q,
        eps=args.eps,
        use_smooth=bool(args.rx_use_smooth),
        smooth_p=float(args.rx_smooth_p),
        exclude_output_embedding=True,
    )

    if bool(args.compute_ppl):
        with torch.no_grad():
            ppl_fq_in, nll_fq_in, _ = perplexity(model_q, eval_loader_in, args.device)
        results["ppl_fakequant_in"] = float(ppl_fq_in)
        results["nll_fakequant_in"] = float(nll_fq_in)
        print(f"[FAKEQUANT] in_nll={nll_fq_in:.4f} in_ppl={ppl_fq_in:.4f}")

    # Save layerwise averages (same format as before)
    keys = sorted(set(rx_dense.keys()) | set(rx_prefix.keys()) | set(xq_dense.keys()))
    layerwise_out: Dict[int, Dict[str, float]] = {}
    for k in keys:
        layerwise_out[k] = {
            "R(X)": float(rx_dense.get(k, float("nan"))),
            "R(Q(X))": float(rx_prefix.get(k, float("nan"))),
            "||X-Q(X)||/||X||": float(xq_dense.get(k, float("nan"))),
        }

    out_path = args.output_dir / "rx_xq_layerwise_init.json"
    out_path.write_text(json.dumps(layerwise_out, indent=2), encoding="utf-8")
    results["layerwise_path"] = str(out_path)

    # Save distributions (dense + prefix; fp32 + quantized)
    rx_tokens_dict: Dict[str, Any] = {"dense": rx_tokens_dense, "prefix_quant": rx_tokens_prefix}
    rw_channels_dict: Dict[str, Any] = {"fp32": rw_channels_fp, "quantized": rw_channels_q}

    rx_tokens_path = args.output_dir / "rx_tokens_by_layer.json"
    rw_channels_path = args.output_dir / "rw_channels_by_module.json"
    rx_tokens_path.write_text(json.dumps(rx_tokens_dict, indent=2), encoding="utf-8")
    rw_channels_path.write_text(json.dumps(rw_channels_dict, indent=2), encoding="utf-8")
    results["rx_tokens_path"] = str(rx_tokens_path)
    results["rw_channels_path"] = str(rw_channels_path)

    # Optional: embedding huge arrays into results.json (off by default)
    if bool(args.embed_distributions_in_results):
        results["rx_tokens"] = rx_tokens_dict
        results["rw_channels"] = rw_channels_dict

    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[SAVE] Wrote: {out_path}")
    print(f"[SAVE] Wrote: {rx_tokens_path}")
    print(f"[SAVE] Wrote: {rw_channels_path}")
    print(f"[SAVE] Wrote: {args.output_dir / 'results.json'}")

    _release_cuda_models(model_q)


if __name__ == "__main__":
    main()
