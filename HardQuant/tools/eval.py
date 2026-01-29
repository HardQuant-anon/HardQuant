from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gpt2.modeling_gpt2 import Conv1D

# Use the exact quantizers + shared iterators from fake_quant.py
from fake_quant import (  # type: ignore
    quantize_activation_per_token_absmax,
    quantize_activation_per_tensor_absmax,
    iter_activation_quant_modules,
    make_activation_quant_fn,
)

PARALLEL_LINEAR_NAMES = {"ColumnParallelLinear", "RowParallelLinear"}


def _is_parallel_linear(module: nn.Module) -> bool:
    return module.__class__.__name__ in PARALLEL_LINEAR_NAMES and hasattr(module, "weight")


def _is_linear_like(module: nn.Module) -> bool:
    return isinstance(module, nn.Linear) or _is_parallel_linear(module)


def _pick_text_column(ds):
    for name in ("text", "content", "raw", "document"):
        if name in ds.column_names:
            return name
    for name in ds.column_names:
        try:
            if getattr(ds.features[name], "dtype", None) == "string":
                return name
        except Exception:
            pass
    raise KeyError("No suitable text column found.")


def _fallback_load_dataset(dataset_tuple: Tuple[str, ...], split: str):
    if len(dataset_tuple) == 2:
        name, subset = dataset_tuple
    elif len(dataset_tuple) == 1:
        (name,) = dataset_tuple
        subset = None
    else:
        raise ValueError(f"Unexpected dataset tuple: {dataset_tuple}")
    try:
        return load_dataset(name, subset, split=split)
    except ValueError as exc:
        if split != "train":
            print(f"[WARN] dataset {dataset_tuple} does not provide split '{split}'. Falling back to 'train'.")
            return load_dataset(name, subset, split="train")
        raise exc


def _format_dolly(example):
    parts = []
    inst = (example.get("instruction") or "").strip()
    if inst:
        parts.append(f"Instruction:\n{inst}")
    ctx = (example.get("context") or "").strip()
    if ctx:
        parts.append(f"Context:\n{ctx}")
    resp = (example.get("response") or "").strip()
    if resp:
        parts.append(f"Response:\n{resp}")
    text = "\n\n".join(parts).strip()
    if not text:
        text = inst or resp or ""
    return {"text": text}


def concat_token_blocks_split(
    tok,
    dataset_tuple: Tuple[str, Optional[str]],
    split: str,
    block_size: int,
    max_tokens: Optional[int] = None,
):
    """
    Tokenize a split and return a (N_blocks, block_size) tensor.
    """
    ds = _fallback_load_dataset(dataset_tuple, split)
    if dataset_tuple[0] == "databricks/databricks-dolly-15k":
        ds = ds.map(_format_dolly)

    text_col = _pick_text_column(ds)
    tok_ds = ds.map(lambda ex: tok(ex[text_col]), batched=True, remove_columns=ds.column_names)

    all_ids = []
    for ids in tok_ds["input_ids"]:
        all_ids.extend(ids)
        if max_tokens is not None and len(all_ids) >= max_tokens + block_size:
            break

    n_full = (len(all_ids) // block_size) * block_size
    if n_full == 0:
        raise RuntimeError(f"not enough tokens for one block (dataset={dataset_tuple}, split={split})")

    ids = torch.tensor(all_ids[:n_full], dtype=torch.long)
    return ids.view(-1, block_size)


def build_eval_loader(
    tok,
    dataset_tuple: Tuple[str, Optional[str]],
    block_size: int,
    batch_size_eval: int,
    split: str,
    max_fraction: Optional[float] = None,
):
    """
    Build an evaluation DataLoader.

    If max_fraction is provided (0 < max_fraction < 1), keep only that
    fraction of the available blocks (rounded down, but at least 1 block).
    """
    blocks = concat_token_blocks_split(tok, dataset_tuple, split, block_size, None)
    nblocks_total = blocks.size(0)

    if max_fraction is not None and 0.0 < max_fraction < 1.0:
        n_keep = max(1, int(nblocks_total * max_fraction))
        blocks = blocks[:n_keep]

    return (
        DataLoader(TensorDataset(blocks), batch_size=batch_size_eval, shuffle=False),
        blocks.size(0),
        blocks.numel(),
    )


def build_train_loader(
    tok,
    dataset_tuple: Tuple[str, Optional[str]],
    block_size: int,
    calib_tokens_train: int,
    batch_size_ft: int,
    split: str = "train",
):
    blocks = concat_token_blocks_split(tok, dataset_tuple, split, block_size, calib_tokens_train)
    return DataLoader(TensorDataset(blocks), batch_size=batch_size_ft, shuffle=True, drop_last=True)


# -------------------- Core eval metrics --------------------
@torch.no_grad()
def perplexity(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    tot_nll, tot_tok = 0.0, 0
    for (x,) in loader:
        x = x.to(device)
        out = model(input_ids=x, labels=x)
        tot_nll += float(out.loss) * x.numel()
        tot_tok += x.numel()
    avg_nll = tot_nll / max(1, tot_tok)
    return math.exp(avg_nll), avg_nll, tot_tok


@torch.no_grad()
def kl_fp32_vs_quant(
    model_ref: nn.Module,
    model_q: nn.Module,
    loader: DataLoader,
    device: str,
) -> float:
    model_ref.eval()
    model_q.eval()
    tot_kl, tot_tok = 0.0, 0

    for (x,) in loader:
        x = x.to(device)

        out_ref = model_ref(input_ids=x)
        out_q = model_q(input_ids=x)

        logits_ref = out_ref.logits
        logits_q = out_q.logits

        logp_ref = F.log_softmax(logits_ref, dim=-1)
        logp_q = F.log_softmax(logits_q, dim=-1)
        p_ref = logp_ref.exp()

        kl_pos = (p_ref * (logp_ref - logp_q)).sum(dim=-1)  # [B, T]
        tot_kl += float(kl_pos.sum())
        tot_tok += x.numel()

    return 0.0 if tot_tok == 0 else (tot_kl / tot_tok)


def eval_fp32(
    model_name: str,
    device: str,
    dataset_tuple: Tuple[str, str],
    block_size: int,
    batch_size_eval: int,
    dataset_unseen: Optional[Tuple[str, str]] = None,
):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    loader, nblocks, ntok = build_eval_loader(
        tok,
        dataset_tuple,
        block_size,
        batch_size_eval,
        split="validation" if dataset_unseen is not None else "test",
    )
    ppl, nll, _ = perplexity(model, loader, device)
    print(f"[FP32] dataset={dataset_tuple} blocks={nblocks} tokens={ntok}  NLL={nll:.6f}  PPL={ppl:.4f}")
    return ppl, model, tok, loader, nblocks, ntok


# -------------------- Quantization helpers for diagnostics --------------------
@dataclass(frozen=True)
class QuantParams:
    bits_w: int = 8
    bits_x: int = 8
    eps: float = 1e-12

    @property
    def QW(self) -> int:
        return (1 << (self.bits_w - 1)) - 1

    @property
    def QX(self) -> int:
        return (1 << (self.bits_x - 1)) - 1


@torch.no_grad()
def qdq_weights_per_column_linear(W: torch.Tensor, qp: QuantParams) -> torch.Tensor:
    Delta = W.abs().amax(dim=1, keepdim=True).clamp_min(qp.eps) / qp.QW
    q = torch.clamp((W / Delta).round(), -qp.QW, qp.QW)
    return (q * Delta).to(W.dtype)


@torch.no_grad()
def qdq_weights_per_column_conv1d(W: torch.Tensor, qp: QuantParams) -> torch.Tensor:
    Delta = W.abs().amax(dim=0, keepdim=True).clamp_min(qp.eps) / qp.QW
    q = torch.clamp((W / Delta).round(), -qp.QW, qp.QW)
    return (q * Delta).to(W.dtype)


@torch.no_grad()
def qdq_weights_per_column_embedding(W: torch.Tensor, qp: QuantParams) -> torch.Tensor:
    Delta = W.abs().amax(dim=0, keepdim=True).clamp_min(qp.eps) / qp.QW
    q = torch.clamp((W / Delta).round(), -qp.QW, qp.QW)
    return (q * Delta).to(W.dtype)


@torch.no_grad()
def qdq_tokenwise(x: torch.Tensor, qp: QuantParams) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        return x
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return x

    if x.dim() == 3:
        s_bt = x.abs().amax(dim=-1)  # (B, T)
        s_t = s_bt.amax(dim=0, keepdim=True)  # (1, T)
        Delta = s_t.clamp_min(qp.eps).unsqueeze(-1) / qp.QX  # (1, T, 1)
    elif x.dim() == 2:
        Delta = x.abs().amax(dim=-1, keepdim=True).clamp_min(qp.eps) / qp.QX  # (N,1)
    else:
        Delta = x.abs().amax(dim=-1, keepdim=True).clamp_min(qp.eps) / qp.QX

    q = torch.clamp((x / Delta).round(), -qp.QX, qp.QX)
    return (q * Delta).to(x.dtype)


@torch.no_grad()
def mean_weight_qerror_across_layers(model: nn.Module, qp: QuantParams) -> float:
    norms, n = 0.0, 0
    for m in model.modules():
        if _is_linear_like(m):
            W, Wq = m.weight.data, qdq_weights_per_column_linear(m.weight.data, qp)
        elif isinstance(m, Conv1D):
            W, Wq = m.weight.data, qdq_weights_per_column_conv1d(m.weight.data, qp)
        elif isinstance(m, nn.Embedding):
            W, Wq = m.weight.data, qdq_weights_per_column_embedding(m.weight.data, qp)
        else:
            continue
        norms += float((W - Wq).float().norm().item())
        n += 1
    return 0.0 if n == 0 else (norms / n)


# -----------------------------------------------------------------------------
# Activation qerror on EXACT activations fake_quant hooks target (shared iterator)
# -----------------------------------------------------------------------------
@torch.no_grad()
def mean_activation_qerror_on_loader_quantized_acts(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    act_quant: str = "per_token",
    act_bits: int = 8,
) -> float:
    """
    Mean over modules that fake_quant would hook (same iterator) of:

        mean_row( ||X - Q(X)||_2 / ||X||_2 )

    where X is the *output* activation of those modules, flattened to [-1, hidden].
    Uses the exact same absmax quantizers as fake_quant.py.
    """
    eps = 1e-12
    act_fn = make_activation_quant_fn(act_quant, act_bits)

    sums, cnts = 0.0, 0
    handles = []

    def hook(_m, _inp, out):
        nonlocal sums, cnts

        if isinstance(out, torch.Tensor):
            t = out
        elif isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
            t = out[0]
        else:
            return

        if t.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return
        if t.numel() == 0 or t.dim() < 2:
            return

        X = t.view(-1, t.size(-1))
        Xq = act_fn(X.detach().clone())  # quantizers operate in-place

        Xf = X.float()
        Ef = (Xf - Xq.float())

        num = Ef.norm(dim=-1)
        den = Xf.norm(dim=-1).clamp_min(eps)
        rel = (num / den).mean().item()

        sums += float(rel)
        cnts += 1

    for _name, m in iter_activation_quant_modules(model, exclude_lm_head=True):
        handles.append(m.register_forward_hook(hook))

    model.eval()
    for (x,) in loader:
        x = x.to(device)
        _ = model(input_ids=x, labels=x)

    for h in handles:
        try:
            h.remove()
        except Exception:
            pass

    return 0.0 if cnts == 0 else (sums / cnts)


@torch.no_grad()
def mean_activation_qerror_on_loader_inputs(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    qp: QuantParams,
    act_mods: Tuple[type, ...] = (nn.Linear, Conv1D),
) -> float:
    """
    Legacy: Unbiased MEAN across act_mods of per-token ||X - Q(X)||_2
    measured on the *inputs* to those modules.
    """
    sums, cnts = 0.0, 0
    handles = []

    def pre(_m, args):
        x = args[0] if isinstance(args, (tuple, list)) else args
        if not isinstance(x, torch.Tensor):
            return
        X = x if x.dim() == 2 else x.view(-1, x.size(-1))
        Xq = qdq_tokenwise(X, qp)
        nonlocal sums, cnts
        sums += float((X - Xq).float().norm(dim=-1).mean().item())
        cnts += 1

    for _, m in model.named_modules():
        if isinstance(m, act_mods) or _is_parallel_linear(m):
            handles.append(m.register_forward_pre_hook(pre))

    model.eval()
    for (x,) in loader:
        x = x.to(device)
        _ = model(input_ids=x, labels=x)

    for h in handles:
        try:
            h.remove()
        except Exception:
            pass

    return 0.0 if cnts == 0 else (sums / cnts)
