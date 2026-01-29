#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gc
import inspect
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download  # type: ignore
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# -----------------------------------------------------------------------------
# Optional model-type imports (robust across transformers versions)
# -----------------------------------------------------------------------------
def _try_import(path: str, name: str):
    try:
        mod = __import__(path, fromlist=[name])
        return getattr(mod, name)
    except Exception:
        return None


OPTPreTrainedModel = _try_import("transformers.models.opt.modeling_opt", "OPTPreTrainedModel")
LlamaPreTrainedModel = _try_import("transformers.models.llama.modeling_llama", "LlamaPreTrainedModel")
MistralPreTrainedModel = _try_import("transformers.models.mistral.modeling_mistral", "MistralPreTrainedModel")
MixtralPreTrainedModel = _try_import("transformers.models.mixtral.modeling_mixtral", "MixtralPreTrainedModel")

# Qwen (non-MoE)
Qwen2PreTrainedModel = _try_import("transformers.models.qwen2.modeling_qwen2", "Qwen2PreTrainedModel")
Qwen3PreTrainedModel = _try_import("transformers.models.qwen3.modeling_qwen3", "Qwen3PreTrainedModel")


# -----------------------------------------------------------------------------
# Resolve model dir
# -----------------------------------------------------------------------------
def _resolve_model_dir(model_dir_or_id: Union[str, Path], local_files_only: bool) -> Path:
    s = str(model_dir_or_id)
    p = Path(s)
    if p.exists():
        return p.resolve()
    local_path = snapshot_download(repo_id=s, local_files_only=bool(local_files_only))
    return Path(local_path).resolve()


# -----------------------------------------------------------------------------
# Project paths and imports
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]  
TOOLS_DIR = PROJECT_DIR / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from eval import build_eval_loader, perplexity  # type: ignore
from fake_quant import add_activation_quant_hooks, quantize_model as fake_quantize_model  # type: ignore

WORKSPACE_DIR = Path("/workspace").expanduser().resolve()

GPTQ_DIR = WORKSPACE_DIR / "gptq-main"
if GPTQ_DIR.exists() and str(GPTQ_DIR) not in sys.path:
    # Put after tools/ to avoid shadowing your tools modules
    sys.path.insert(1, str(GPTQ_DIR))

from gptq import GPTQ as GPTQLayer  # type: ignore
from modelutils import find_layers as gptq_find_layers  # type: ignore
from quant import Quantizer as GPTQQuantizer  # type: ignore

# OmniQuant repo root (for imports)
OMNIQUANT_DIR = WORKSPACE_DIR / "OmniQuant-main"
if OMNIQUANT_DIR.exists() and str(OMNIQUANT_DIR) not in sys.path:
    # Put after tools/ and gptq-main to reduce collisions
    sys.path.insert(2, str(OMNIQUANT_DIR))


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
DEFAULT_EVAL_DATASETS = [
    "wikitext:wikitext-2-raw-v1::validation",
    "wikitext:wikitext-2-raw-v1::test",
    "lambada:plain_text::validation",
    "stas/openwebtext-10k::train[:500]",
    "brando/small-c4-dataset::test[:500]",
]


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _release_cuda_models(*models: object) -> None:
    for m in models:
        try:
            del m
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _remove_hooks(hooks: List[Any]) -> None:
    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass


def _model_type(model: nn.Module) -> Optional[str]:
    return getattr(getattr(model, "config", None), "model_type", None)


def _is_supported_causallm(model: nn.Module) -> bool:
    for cls in (
        OPTPreTrainedModel,
        LlamaPreTrainedModel,
        MistralPreTrainedModel,
        MixtralPreTrainedModel,
        Qwen2PreTrainedModel,
        Qwen3PreTrainedModel,
    ):
        if cls is not None and isinstance(model, cls):
            return True

    mt = _model_type(model)
    return mt in {
        "opt",
        "llama",
        "mistral",
        "mixtral",
        "qwen2",
        "qwen3",
    }


def _looks_llama_like_decoder(model: nn.Module) -> bool:
    m = getattr(model, "model", None)
    return bool(m is not None and hasattr(m, "layers") and hasattr(m, "embed_tokens"))


def parse_dataset_name(arg: str) -> Tuple[str, Optional[str]]:
    if ":" in arg:
        name, subset = arg.split(":", 1)
        subset = subset.strip()
        return name.strip(), (subset if subset else None)
    return arg.strip(), None


def parse_eval_spec(spec: str, default_split: str) -> Tuple[Tuple[str, Optional[str]], str]:
    # accept "ds:subset::split" and "ds:subset@split"
    if "::" in spec:
        ds_part, split = spec.rsplit("::", 1)
        return parse_dataset_name(ds_part.strip()), split.strip()
    if "@" in spec:
        ds_part, split = spec.rsplit("@", 1)
        return parse_dataset_name(ds_part.strip()), split.strip()
    return parse_dataset_name(spec.strip()), default_split


def hf_tuple(dataset: Tuple[str, Optional[str]]) -> Tuple[str, ...]:
    return (dataset[0], dataset[1]) if dataset[1] is not None else (dataset[0],)


def _make_tokenizer(model_dir_or_id: str):
    """
    Qwen tokenizers sometimes need trust_remote_code depending on the env.
    We try native first, then fall back.
    """
    last_err: Optional[Exception] = None
    tok = None
    for kw in (
        dict(use_fast=True, trust_remote_code=False),
        dict(use_fast=True, trust_remote_code=True),
        dict(use_fast=False, trust_remote_code=True),
    ):
        try:
            tok = AutoTokenizer.from_pretrained(model_dir_or_id, **kw)
            break
        except Exception as e:
            last_err = e
            tok = None
    if tok is None:
        raise RuntimeError(f"Failed to load tokenizer for {model_dir_or_id}. Last error: {last_err}") from last_err

    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    return tok


def _resolve_dtype(dtype_str: str):
    s = (dtype_str or "auto").lower().strip()
    if s == "auto":
        return None
    if s in {"fp32", "float32"}:
        return torch.float32
    if s in {"fp16", "float16"}:
        return torch.float16
    if s in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"Unknown --dtype {dtype_str}. Use auto|fp32|fp16|bf16.")


@dataclass
class EvalConfig:
    block_size: int
    batch_size_eval: int
    eval_fraction: float


def _validate_fraction(frac: float) -> float:
    f = float(frac)
    if not (0.0 < f <= 1.0):
        raise ValueError(f"eval_fraction must be in (0,1], got {frac}")
    return f


# -----------------------------------------------------------------------------
# Rotary helpers (newer transformers may require position_embeddings)
# -----------------------------------------------------------------------------
def _get_model_rotary_emb(model: nn.Module):
    m = getattr(model, "model", None)
    return getattr(m, "rotary_emb", None) if m is not None else None


def _force_move_rotary_emb_to(rotary_emb: Optional[nn.Module], device) -> None:
    if rotary_emb is None:
        return
    rotary_emb.to(device)
    # buffers + common cached tensors
    for k, v in list(getattr(rotary_emb, "_buffers", {}).items()):
        if torch.is_tensor(v):
            rotary_emb._buffers[k] = v.to(device)
    for attr in (
        "inv_freq",
        "inv_freq_expanded",
        "cos_cached",
        "sin_cached",
        "_cos_cached",
        "_sin_cached",
        "cos",
        "sin",
    ):
        if hasattr(rotary_emb, attr):
            t = getattr(rotary_emb, attr)
            if torch.is_tensor(t):
                setattr(rotary_emb, attr, t.to(device))


def _ensure_model_rotary_on_device(model: nn.Module, device) -> None:
    _force_move_rotary_emb_to(_get_model_rotary_emb(model), device)


def _layer_needs_position_embeddings(layer: nn.Module) -> bool:
    cached = getattr(layer, "_needs_position_embeddings", None)
    if isinstance(cached, bool):
        return cached
    needs = False
    try:
        needs = "position_embeddings" in inspect.signature(layer.forward).parameters
    except Exception:
        needs = False
    setattr(layer, "_needs_position_embeddings", needs)
    return needs


# -----------------------------------------------------------------------------
# OmniQuant <-> transformers kwarg compatibility (CRITICAL FIX)
# -----------------------------------------------------------------------------
def _patch_decoder_layers_accept_past_key_values(model: nn.Module) -> int:
    """
    Patch OmniQuant QuantLlamaDecoderLayer.forward to be compatible with newer transformers:
      - drops unexpected kwargs (past_key_values, cache_position, etc.)
      - unwraps hidden_states if it arrives as (tensor,)
      - normalizes return so output[0] is always a Tensor (never (tensor,))
    """
    patched_classes = 0
    seen: set[type] = set()

    def _unwrap_tensor_tuple(x: Any) -> Any:
        # Common failure mode: hidden_states arrives as (tensor,)
        if isinstance(x, tuple) and len(x) == 1 and torch.is_tensor(x[0]):
            return x[0]
        return x

    def _normalize_layer_output(out: Any) -> Any:
        # Transformers expects tuple-like where [0] is hidden_states Tensor.
        # OmniQuant variants sometimes return ((hs,), ...) -> fix to (hs, ...)
        if torch.is_tensor(out):
            return (out,)

        if isinstance(out, (list, tuple)) and len(out) > 0:
            first = out[0]
            first = _unwrap_tensor_tuple(first)
            if first is not out[0]:
                if isinstance(out, tuple):
                    return (first,) + tuple(out[1:])
                else:
                    return [first] + list(out[1:])
        return out

    for m in model.modules():
        cls = m.__class__
        cname = cls.__name__

        # Most precise match for OmniQuant
        if cname != "QuantLlamaDecoderLayer":
            continue

        if cls in seen:
            continue
        seen.add(cls)

        if getattr(cls, "_hf_kwarg_compat_patched", False):
            continue

        orig_forward = cls.forward

        def new_forward(self, *args, **kwargs):
            # Unwrap hidden_states if it is (tensor,)
            if len(args) >= 1:
                a0 = _unwrap_tensor_tuple(args[0])
                if a0 is not args[0]:
                    args = (a0,) + tuple(args[1:])

            # Translate common naming mismatch
            if "past_key_values" in kwargs and "past_key_value" not in kwargs:
                kwargs["past_key_value"] = kwargs.pop("past_key_values")

            # Also unwrap if someone passed hidden_states via kw (rare)
            if "hidden_states" in kwargs:
                kwargs["hidden_states"] = _unwrap_tensor_tuple(kwargs["hidden_states"])

            # Retry loop: drop any unexpected kwargs that transformers adds
            while True:
                try:
                    out = orig_forward(self, *args, **kwargs)
                    return _normalize_layer_output(out)
                except TypeError as e:
                    msg = str(e)
                    if "unexpected keyword argument" not in msg:
                        raise

                    # Extract kw name from "... 'cache_position'" etc.
                    kw = None
                    if "'" in msg:
                        parts = msg.split("'")
                        if len(parts) >= 2:
                            kw = parts[1]

                    if not kw or kw not in kwargs:
                        raise

                    kwargs.pop(kw, None)

        cls.forward = new_forward  # type: ignore[assignment]
        setattr(cls, "_hf_kwarg_compat_patched", True)
        patched_classes += 1

    return patched_classes


# -----------------------------------------------------------------------------
# OmniQuant <-> transformers kwarg compatibility (CRITICAL FIX)
# -----------------------------------------------------------------------------
def _patch_omniquant_decoder_layers_hf_compat(model: nn.Module) -> Dict[str, int]:
    """
    Patch OmniQuant decoder layers to be compatible with newer HF Transformers that may pass:
      - past_key_values
      - cache_position
      - position_embeddings
      - etc.

    Also normalizes return so output[0] is a Tensor (never (Tensor,)).
    """
    targets = {
        "QuantLlamaDecoderLayer",
        "QuantQwenDecoderLayer",
        "QuantQwen3DecoderLayer",
    }

    patched_by_class = 0
    patched_instances = 0
    seen: set[type] = set()

    def _unwrap_tensor_tuple(x: Any) -> Any:
        if isinstance(x, (tuple, list)) and len(x) == 1 and torch.is_tensor(x[0]):
            return x[0]
        return x

    def _normalize_layer_output(out: Any) -> Any:
        # Ensure tuple-like and out[0] is a Tensor, not (Tensor,)
        if torch.is_tensor(out):
            return (out,)
        if isinstance(out, (tuple, list)) and len(out) > 0:
            first = _unwrap_tensor_tuple(out[0])
            if first is not out[0]:
                if isinstance(out, tuple):
                    return (first,) + tuple(out[1:])
                return [first] + list(out[1:])
        return out

    for m in model.modules():
        cls = m.__class__
        cname = cls.__name__
        if cname not in targets:
            continue

        patched_instances += 1

        if cls in seen:
            continue
        seen.add(cls)

        if getattr(cls, "_hf_kwarg_compat_patched", False):
            continue

        orig_forward = cls.forward

        def new_forward(self, *args, **kwargs):
            # hidden_states sometimes leaks as (tensor,)
            if len(args) >= 1:
                a0 = _unwrap_tensor_tuple(args[0])
                if a0 is not args[0]:
                    args = (a0,) + tuple(args[1:])
            if "hidden_states" in kwargs:
                kwargs["hidden_states"] = _unwrap_tensor_tuple(kwargs["hidden_states"])

            # HF uses past_key_values; OmniQuant often uses past_key_value
            if "past_key_values" in kwargs and "past_key_value" not in kwargs:
                kwargs["past_key_value"] = kwargs.pop("past_key_values")

            # Retry loop: drop any unexpected kwargs HF introduces
            while True:
                try:
                    out = orig_forward(self, *args, **kwargs)
                    return _normalize_layer_output(out)
                except TypeError as e:
                    msg = str(e)
                    if "unexpected keyword argument" not in msg:
                        raise

                    bad_kw = None
                    if "'" in msg:
                        parts = msg.split("'")
                        if len(parts) >= 2:
                            bad_kw = parts[1]

                    if not bad_kw or bad_kw not in kwargs:
                        raise

                    kwargs.pop(bad_kw, None)

        cls.forward = new_forward  # type: ignore[assignment]
        setattr(cls, "_hf_kwarg_compat_patched", True)
        patched_by_class += 1

    return {
        "patched_classes": int(patched_by_class),
        "seen_instances": int(patched_instances),
    }


def _patch_hf_qwen3_norm_unwrap() -> int:
    """
    Patch transformers' Qwen3 norm forward to accept tuple/list hidden_states.

    Fixes:
      AttributeError: 'tuple' object has no attribute 'dtype'
    when hidden_states leaks as (tensor,) into HF Qwen3 norms (including final model norm).
    """
    try:
        import transformers.models.qwen3.modeling_qwen3 as qwen3_mod  # type: ignore
    except Exception:
        return 0

    # Qwen3 norm class name has been stable in HF releases as Qwen3RMSNorm,
    # but keep it defensive in case of small renames.
    NormCls = getattr(qwen3_mod, "Qwen3RMSNorm", None)
    if NormCls is None:
        # fallback: scan for a class that looks like the norm
        for name in ("RMSNorm", "QwenRMSNorm", "Qwen3Norm", "Qwen3LayerNorm"):
            NormCls = getattr(qwen3_mod, name, None)
            if NormCls is not None:
                break
    if NormCls is None:
        return 0

    if getattr(NormCls, "_omniquant_unwrap_patched", False):
        return 0

    orig_forward = NormCls.forward

    def new_forward(self, hidden_states, *args, **kwargs):
        if isinstance(hidden_states, (tuple, list)) and len(hidden_states) > 0 and torch.is_tensor(hidden_states[0]):
            hidden_states = hidden_states[0]
        while isinstance(hidden_states, (tuple, list)) and len(hidden_states) == 1:
            inner = hidden_states[0]
            if torch.is_tensor(inner):
                hidden_states = inner
                break
            if isinstance(inner, (tuple, list)) and len(inner) > 0 and torch.is_tensor(inner[0]):
                hidden_states = inner[0]
                break
            hidden_states = inner
        return orig_forward(self, hidden_states, *args, **kwargs)

    NormCls.forward = new_forward  # type: ignore[assignment]
    setattr(NormCls, "_omniquant_unwrap_patched", True)
    return 1


def _patch_hf_llama_rmsnorm_unwrap() -> int:
    """
    Patch transformers' LlamaRMSNorm.forward to accept tuple/list hidden_states.

    Fixes:
      AttributeError: 'tuple' object has no attribute 'dtype'
    when hidden_states leaks as (tensor,) into HF norms (including final model norm).
    """
    try:
        from transformers.models.llama.modeling_llama import LlamaRMSNorm  # type: ignore
    except Exception:
        return 0

    if getattr(LlamaRMSNorm, "_omniquant_unwrap_patched", False):
        return 0

    orig_forward = LlamaRMSNorm.forward

    def new_forward(self, hidden_states, *args, **kwargs):
        if isinstance(hidden_states, (tuple, list)) and len(hidden_states) > 0 and torch.is_tensor(hidden_states[0]):
            hidden_states = hidden_states[0]
        while isinstance(hidden_states, (tuple, list)) and len(hidden_states) == 1:
            inner = hidden_states[0]
            if torch.is_tensor(inner):
                hidden_states = inner
                break
            if isinstance(inner, (tuple, list)) and len(inner) > 0 and torch.is_tensor(inner[0]):
                hidden_states = inner[0]
                break
            hidden_states = inner
        return orig_forward(self, hidden_states, *args, **kwargs)

    LlamaRMSNorm.forward = new_forward  # type: ignore[assignment]
    setattr(LlamaRMSNorm, "_omniquant_unwrap_patched", True)
    return 1


def _decoder_layer_forward_with_optional_position_embeddings(
    model: nn.Module,
    layer: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
):
    kwargs: Dict[str, Any] = {}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    if position_ids is not None:
        kwargs["position_ids"] = position_ids

    if _layer_needs_position_embeddings(layer) and position_ids is not None:
        rotary = _get_model_rotary_emb(model) or getattr(getattr(layer, "self_attn", None), "rotary_emb", None)
        if rotary is not None:
            _force_move_rotary_emb_to(rotary, hidden_states.device)
            pos_emb = rotary(hidden_states, position_ids)
            if pos_emb is not None:
                kwargs["position_embeddings"] = pos_emb

    return layer(hidden_states, **kwargs)[0]


def _extract_input_ids_and_attention_mask(batch: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if isinstance(batch, dict):
        x = batch.get("input_ids", None)
        if x is None:
            raise RuntimeError("Batch dict missing 'input_ids'.")
        am = batch.get("attention_mask", None)
        return x, (am if torch.is_tensor(am) else None)

    if isinstance(batch, (tuple, list)):
        if not batch:
            raise RuntimeError("Empty batch.")
        x = batch[0]
        am = batch[1] if (len(batch) > 1 and torch.is_tensor(batch[1])) else None
        return x, am

    if torch.is_tensor(batch):
        return batch, None

    raise RuntimeError(f"Unrecognized batch type: {type(batch)}")


# -----------------------------------------------------------------------------
# GPTQ
# -----------------------------------------------------------------------------
@torch.no_grad()
def gptq_quantize_weights_inplace_llama_like(
    model: nn.Module,
    tok,
    device: str,
    calib_spec: str,
    default_split: str,
    nsamples: int,
    seqlen: int,
    wbits: int,
    groupsize: int,
    percdamp: float,
    act_order: bool,
    true_sequential: bool,
    static_groups: bool,
    sym: bool,
) -> Dict[str, Any]:
    if not _looks_llama_like_decoder(model):
        raise RuntimeError(
            "GPTQ weights path supports LLaMA/Mistral/Mixtral-style decoder models only (model.model.layers)."
        )

    dev = torch.device(device)
    model.eval()

    use_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = False

    layers = model.model.layers

    (calib_ds, calib_split) = parse_eval_spec(calib_spec, default_split=default_split)
    calib_loader, _nblocks, _ntok = build_eval_loader(
        tok,
        hf_tuple(calib_ds),
        int(seqlen),  # block_size
        1,  # batch_size_eval
        split=calib_split,
        max_fraction=1.0,
    )

    print(f"[GPTQ] calib={calib_spec} nsamples={nsamples} seqlen={seqlen} wbits={wbits} device={device}")
    print("[GPTQ] Catcher pass ...")

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    if getattr(model.model, "norm", None) is not None:
        model.model.norm = model.model.norm.to(dev)
    _ensure_model_rotary_on_device(model, dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, seqlen, int(model.config.hidden_size)), dtype=dtype, device=dev)
    cache: Dict[str, Any] = {"i": 0, "attention_mask": None, "position_ids": None}

    class Catcher(nn.Module):
        """
        Wrap a decoder layer to capture its input hidden_states during the first layer pass.
        """

        def __init__(self, module: nn.Module):
            super().__init__()
            self.module = module
            if hasattr(module, "attention_type"):
                self.attention_type = getattr(module, "attention_type")

        def __getattr__(self, name: str):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

        def forward(self, inp, **kwargs):
            i = int(cache["i"])
            if i >= nsamples:
                raise ValueError
            inps[i] = inp
            cache["i"] = i + 1
            if "attention_mask" in kwargs:
                cache["attention_mask"] = kwargs["attention_mask"]
            if "position_ids" in kwargs:
                cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])

    for batch in calib_loader:
        if int(cache["i"]) >= nsamples:
            break

        input_ids, attn = _extract_input_ids_and_attention_mask(batch)
        input_ids = input_ids.to(dev)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        if input_ids.size(-1) < seqlen:
            continue
        if input_ids.size(-1) > seqlen:
            input_ids = input_ids[:, :seqlen]

        kwargs: Dict[str, Any] = {}
        if attn is not None:
            attn = attn.to(dev)
            if attn.dim() == 1:
                attn = attn.unsqueeze(0)
            if attn.size(-1) >= seqlen:
                attn = attn[:, :seqlen]
            kwargs["attention_mask"] = attn

        try:
            _ensure_model_rotary_on_device(model, dev)
            model(input_ids, **kwargs)
        except ValueError:
            pass

    layers[0] = layers[0].module

    got = int(cache["i"])
    if got < nsamples:
        raise RuntimeError(f"[GPTQ] Only collected {got} calibration sequences; need nsamples={nsamples}.")

    attention_mask = cache.get("attention_mask", None)
    position_ids = cache.get("position_ids", None)
    if position_ids is None:
        position_ids = torch.arange(seqlen, device=dev).unsqueeze(0)

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if getattr(model.model, "norm", None) is not None:
        model.model.norm = model.model.norm.cpu()
    _ensure_model_rotary_on_device(model, dev)
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)

    print("[GPTQ] Quantizing weights ...")

    n_modules_quantized = 0
    for li in range(len(layers)):
        layer = layers[li].to(dev)
        full = gptq_find_layers(layer)

        sequential = (
            [
                ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
                ["self_attn.o_proj"],
                ["mlp.up_proj", "mlp.gate_proj"],
                ["mlp.down_proj"],
            ]
            if true_sequential
            else [list(full.keys())]
        )

        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq_objs: Dict[str, Any] = {}
            for name, mod in subset.items():
                g = GPTQLayer(mod)
                q = GPTQQuantizer()
                q.configure(int(wbits), perchannel=True, sym=bool(sym), mse=False)
                g.quantizer = q
                gptq_objs[name] = g

            def add_batch(name: str):
                def tmp(_, inp, out):
                    gptq_objs[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = [subset[name].register_forward_hook(add_batch(name)) for name in subset]

            for j in range(nsamples):
                outs[j] = _decoder_layer_forward_with_optional_position_embeddings(
                    model=model,
                    layer=layer,
                    hidden_states=inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )

            for h in handles:
                h.remove()

            for name in subset:
                print(f"[GPTQ] layer={li} module={name}")
                gptq_objs[name].fasterquant(
                    percdamp=float(percdamp),
                    groupsize=int(groupsize),
                    actorder=bool(act_order),
                    static_groups=bool(static_groups),
                )
                gptq_objs[name].free()
                n_modules_quantized += 1

        for j in range(nsamples):
            outs[j] = _decoder_layer_forward_with_optional_position_embeddings(
                model=model,
                layer=layer,
                hidden_states=inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

        layers[li] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache

    return {
        "calib_spec": calib_spec,
        "calib_split": calib_split,
        "nsamples": int(nsamples),
        "seqlen": int(seqlen),
        "wbits": int(wbits),
        "groupsize": int(groupsize),
        "percdamp": float(percdamp),
        "act_order": bool(act_order),
        "true_sequential": bool(true_sequential),
        "static_groups": bool(static_groups),
        "sym": bool(sym),
        "n_modules_quantized": int(n_modules_quantized),
    }


# -----------------------------------------------------------------------------
# SmoothQuant integration
# -----------------------------------------------------------------------------
def _default_smoothquant_root() -> Path:
    return (PROJECT_DIR.parent / "sft-smoothquant").resolve()


def _smoothquant_import(sq_root: Path):
    """
    Import smoothquant.smooth.smooth_lm and smoothquant.fake_quant.quantize_model
    from the SmoothQuant repo used by sft-smoothquant/run_sft_smoothquant.py.

    sq_root should be the directory that CONTAINS the `smoothquant/` package directory.
    i.e. sq_root/.../smoothquant
    """
    sq_root = sq_root.resolve()
    sq_pkg_dir = sq_root / "smoothquant"
    if not sq_pkg_dir.exists():
        raise FileNotFoundError(f"SmoothQuant package dir not found: {sq_pkg_dir}")

    if str(sq_root) not in sys.path:
        sys.path.insert(3, str(sq_root))

    from smoothquant.smooth import smooth_lm  # type: ignore
    from smoothquant.fake_quant import quantize_model as sq_quantize_model  # type: ignore

    return smooth_lm, sq_quantize_model, sq_root, sq_pkg_dir


def _run_smoothquant_generate_act_scales(
    *,
    sq_root: Path,
    sq_pkg_dir: Path,
    model_name_or_dir: str,
    out_scales_path: Path,
    dataset_path: Path,
    dataset_name: str,
    dataset_subset: Optional[str],
    dataset_split: str,
    num_samples: int,
    seq_len: int,
    device_map: str,
    extra_env: Optional[Dict[str, str]] = None,
) -> None:
    script = sq_pkg_dir / "examples" / "generate_act_scales.py"
    if not script.exists():
        raise FileNotFoundError(f"SmoothQuant generate_act_scales.py not found at: {script}")

    out_scales_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)

    env["PYTHONPATH"] = f"{sq_root}{os.pathsep}{env.get('PYTHONPATH', '')}"

    cmd = [
        "python",
        str(script),
        "--model-name",
        str(model_name_or_dir),
        "--output-path",
        str(out_scales_path),
        "--num-samples",
        str(int(num_samples)),
        "--seq-len",
        str(int(seq_len)),
        "--dataset-path",
        str(dataset_path),
        "--dataset-name",
        str(dataset_name),
        "--dataset-split",
        str(dataset_split),
        "--device-map",
        str(device_map),
    ]
    if dataset_subset is not None and str(dataset_subset).strip():
        cmd += ["--dataset-subset", str(dataset_subset)]

    print(f"[SQ] generate_act_scales: {' '.join(cmd)}")
    ret = subprocess.run(cmd, cwd=str(sq_root), env=env)
    if ret.returncode != 0:
        raise RuntimeError(f"SmoothQuant act scale generation failed (rc={ret.returncode}). See command above.")


# -----------------------------------------------------------------------------
# SpinQuant helpers
# -----------------------------------------------------------------------------
def _find_spinquant_dir() -> Optional[Path]:
    candidates: List[Path] = [
        WORKSPACE_DIR / "SpinQuant-main",
        WORKSPACE_DIR / "Quantization" / "SpinQuant-main",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return None


def _run_and_tee(cmd: List[str], cwd: Optional[Path], env: Optional[Dict[str, str]], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[CMD] {' '.join(cmd)}")
    if cwd is not None:
        print(f"[CMD] cwd={cwd}")
    print(f"[CMD] log={log_path}")

    with open(log_path, "w", encoding="utf-8") as f:
        p = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        assert p.stdout is not None
        for line in p.stdout:
            print(line, end="")
            f.write(line)
        rc = p.wait()
        if rc != 0:
            raise RuntimeError(f"Command failed with return code {rc}. See log: {log_path}")


def _spinquant_bool_flag(name: str, val: bool) -> str:
    return f"--{name}" if bool(val) else f"--no-{name}"


def _spinquant_build_args_via_process_args_ptq(
    spinq_dir: Path,
    *,
    input_model: str,
    optimized_rotation_path: Path,
    seed: int,
    w_bits: int,
    a_bits: int,
    model_max_length: int,
    per_device_eval_batch_size: int,
    bf16: bool,
    fp16: bool,
    rotate: Optional[bool] = True,
    rotate_mode: Optional[str] = None,
    fp32_had: Optional[bool] = None,
    w_rtn: Optional[bool] = None,
    a_groupsize: Optional[int] = None,
    w_groupsize: Optional[int] = None,
    percdamp: Optional[float] = None,
    nsamples: Optional[int] = None,
    act_order: Optional[bool] = None,
    load_qmodel_path: Optional[str] = None,
    save_qmodel_path: Optional[str] = None,
    export_to_et: Optional[bool] = None,
    output_dir: Optional[Path] = None,
    logging_dir: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
) -> Tuple[Any, Any, Any]:
    if str(spinq_dir) not in sys.path:
        sys.path.insert(4, str(spinq_dir))

    from utils.process_args import process_args_ptq  # type: ignore

    argv = ["spinquant_ptq", "--seed", str(int(seed))]

    if rotate is not None:
        argv += [_spinquant_bool_flag("rotate", bool(rotate))]
    if rotate_mode is not None:
        argv += ["--rotate_mode", str(rotate_mode)]
    if fp32_had is not None:
        argv += [_spinquant_bool_flag("fp32_had", bool(fp32_had))]
    if w_rtn is not None:
        argv += [_spinquant_bool_flag("w_rtn", bool(w_rtn))]
    if a_groupsize is not None:
        argv += ["--a_groupsize", str(int(a_groupsize))]
    if w_groupsize is not None:
        argv += ["--w_groupsize", str(int(w_groupsize))]
    if percdamp is not None:
        argv += ["--percdamp", str(float(percdamp))]
    if nsamples is not None:
        argv += ["--nsamples", str(int(nsamples))]
    if act_order is not None:
        argv += [_spinquant_bool_flag("act_order", bool(act_order))]
    if load_qmodel_path is not None:
        argv += ["--load_qmodel_path", str(load_qmodel_path)]
    if save_qmodel_path is not None:
        argv += ["--save_qmodel_path", str(save_qmodel_path)]
    if export_to_et is not None:
        argv += [_spinquant_bool_flag("export_to_et", bool(export_to_et))]

    argv += ["--w_bits", str(int(w_bits)), "--a_bits", str(int(a_bits))]
    argv += ["--input_model", str(input_model)]
    argv += ["--optimized_rotation_path", str(optimized_rotation_path)]

    argv += ["--model_max_length", str(int(model_max_length))]
    argv += ["--per_device_eval_batch_size", str(int(per_device_eval_batch_size))]
    if bool(bf16):
        argv += ["--bf16"]
    if bool(fp16):
        argv += ["--fp16"]

    if output_dir is not None:
        argv += ["--output_dir", str(output_dir)]
    if logging_dir is not None:
        argv += ["--logging_dir", str(logging_dir)]
    if cache_dir is not None:
        argv += ["--cache_dir", str(cache_dir)]

    old_argv = sys.argv
    try:
        sys.argv = argv
        model_args, training_args, ptq_args = process_args_ptq()
    finally:
        sys.argv = old_argv

    return model_args, training_args, ptq_args


def _run_spinquant_optimize_rotations(
    spinq_dir: Path,
    model_id_or_dir: str,
    rot_out_dir: Path,
    extra_args: List[str],
    log_path: Path,
) -> None:
    script = spinq_dir / "scripts" / "10_optimize_rotation.sh"
    if not script.exists():
        raise FileNotFoundError(f"SpinQuant optimize script not found at: {script}")

    rot_out_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["INPUT_MODEL"] = str(model_id_or_dir)
    env["MODEL"] = str(model_id_or_dir)
    env["HF_MODEL"] = str(model_id_or_dir)
    env["OUTPUT_DIR"] = str(rot_out_dir)
    env["ROT_DIR"] = str(rot_out_dir)

    cmd = ["bash", str(script)] + list(extra_args)
    _run_and_tee(cmd=cmd, cwd=spinq_dir, env=env, log_path=log_path)


def _find_rotation_file(rot_root: Path) -> Optional[Path]:
    if not rot_root.exists():
        return None

    preferred = [rot_root / "R.bin", rot_root / "R.pt", rot_root / "R.pth", rot_root / "R.npy"]
    for p in preferred:
        if p.exists():
            return p.resolve()

    found: List[Path] = []
    for pat in ("R.bin", "R.pt", "R.pth", "R.npy"):
        found.extend(rot_root.rglob(pat))

    if not found:
        for ext in (".bin", ".pt", ".pth", ".npy"):
            for p in rot_root.rglob(f"*{ext}"):
                name = p.name.lower()
                if "rot" in name or "rotation" in name:
                    found.append(p)

    if not found:
        return None

    found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return found[0].resolve()


def _pick_spinquant_torch_dtype(args, resolved_dtype: Optional[torch.dtype]) -> torch.dtype:
    if bool(args.spinquant_force_bf16) and bool(args.spinquant_force_fp16):
        raise ValueError("Choose only one: --spinquant-force-bf16 or --spinquant-force-fp16")

    if bool(args.spinquant_force_bf16):
        return torch.bfloat16
    if bool(args.spinquant_force_fp16):
        return torch.float16

    if resolved_dtype in (torch.bfloat16, torch.float16):
        return resolved_dtype

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _parse_kv_args(kvs: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for s in kvs:
        if "=" not in s:
            raise ValueError(f"Invalid --spinquant-ptq-kv '{s}'. Expected key=value.")
        k, v = s.split("=", 1)
        k, v = k.strip(), v.strip()
        if not k:
            raise ValueError(f"Invalid --spinquant-ptq-kv '{s}'. Empty key.")

        lv = v.lower()
        if lv in {"true", "false"}:
            out[k] = (lv == "true")
            continue

        try:
            if v.startswith("0") and len(v) > 1 and v.isdigit():
                raise ValueError
            out[k] = int(v)
            continue
        except Exception:
            pass

        try:
            out[k] = float(v)
            continue
        except Exception:
            pass

        out[k] = v
    return out


def _spinquant_import(spinq_dir: Path):
    if str(spinq_dir) not in sys.path:
        sys.path.insert(4, str(spinq_dir))
    from eval_utils.main import ptq_model as spinquant_ptq_model  # type: ignore
    from eval_utils.modeling_llama import LlamaForCausalLM as SpinQuantLlamaForCausalLM  # type: ignore

    return spinquant_ptq_model, SpinQuantLlamaForCausalLM


def _infer_omniquant_net_from_config(cfg: AutoConfig, fallback: str) -> str:
    mt = getattr(cfg, "model_type", None)
    if isinstance(mt, str):
        mt = mt.lower()
        if mt == "opt":
            return "opt-1.3b"
        if mt in {"llama", "mistral", "mixtral"}:
            return "llama-7b"
        if mt in {"qwen2", "qwen3"}:
            return "llama-7b"
        if mt == "falcon":
            return "falcon-7b"
    return fallback


def _ensure_omniquant_on_syspath(omni_dir: Path) -> None:
    omni_dir = omni_dir.resolve()
    if str(omni_dir) not in sys.path:
        sys.path.insert(2, str(omni_dir))


def _omniquant_import(omni_dir: Path):
    _ensure_omniquant_on_syspath(omni_dir)
    from models.LMClass import LMClass  # type: ignore
    from quantize.omniquant import omniquant  # type: ignore
    from datautils import get_loaders  # type: ignore
    from parallel_utils import map_layers_to_multi_gpus, get_lowest_occupied_gpu  # type: ignore
    import utils as omni_utils  # type: ignore

    return LMClass, omniquant, get_loaders, map_layers_to_multi_gpus, get_lowest_occupied_gpu, omni_utils


def _cache_path_for_omni_calib(
    cache_dir: Path,
    model_family: str,
    model_id: str,
    calib_tag: str,
    seqlen: int,
    nsamples: int,
    seed: int,
) -> Path:
    safe_model = model_id.replace("/", "_").replace(":", "_")
    safe_tag = calib_tag.replace("/", "_").replace(":", "_")
    return cache_dir / f"dataloader_{model_family}_{safe_tag}_seqlen{seqlen}_ns{nsamples}_seed{seed}_{safe_model}.pt"


@torch.no_grad()
def _build_wikitext103_calib_loader(model_id: str, seqlen: int, nsamples: int, seed: int):
    # Mirrors OmniQuant main.py: wikitext-103 train -> stream -> slices
    from datasets import load_dataset  # local import to avoid pinning
    from transformers import AutoTokenizer  # local import

    traindata = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

    random.seed(int(seed))
    total_needed = int(nsamples) * int(seqlen)
    if total_needed <= 0:
        return []

    indices = list(range(len(traindata)))
    random.shuffle(indices)

    chunks: List[torch.Tensor] = []
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
    for s in range(int(nsamples)):
        start = s * int(seqlen)
        end = start + int(seqlen)
        inp = stream[start:end].unsqueeze(0)
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader


@torch.no_grad()
def _build_wikitext2_calib_loader(model_id: str, seqlen: int, nsamples: int, seed: int):
    # WikiText-2 train -> stream -> slices (same style as _build_wikitext103_calib_loader)
    from datasets import load_dataset  # local import to avoid pinning
    from transformers import AutoTokenizer  # local import

    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

    random.seed(int(seed))
    total_needed = int(nsamples) * int(seqlen)
    if total_needed <= 0:
        return []

    indices = list(range(len(traindata)))
    random.shuffle(indices)

    chunks: List[torch.Tensor] = []
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
    for s in range(int(nsamples)):
        start = s * int(seqlen)
        end = start + int(seqlen)
        inp = stream[start:end].unsqueeze(0)
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader


def _omniquant_should_deactivate_amp(wbits: int, abits: int) -> bool:
    # same condition as OmniQuant main.py
    return ((wbits < 16 and wbits >= 8) or (abits < 16 and abits >= 8))


@torch.no_grad()
def _omniquant_generate_act_scales_shifts(
    model: nn.Module,
    dataloader: List[Tuple[torch.Tensor, torch.Tensor]],
    num_samples: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Regenerate OmniQuant LET act_scales/act_shifts in-process.
    """
    model.eval()
    device = next(model.parameters()).device

    act_scales: Dict[str, torch.Tensor] = {}
    act_shifts: Dict[str, torch.Tensor] = {}

    def stat_scale(name: str, x: torch.Tensor) -> None:
        hidden_dim = x.shape[-1]
        t = x.view(-1, hidden_dim).abs().detach()
        coming_max = torch.max(t, dim=0)[0].float().cpu()
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], coming_max)
        else:
            act_scales[name] = coming_max

    def stat_shift(name: str, x: torch.Tensor) -> None:
        hidden_dim = x.shape[-1]
        t = x.view(-1, hidden_dim).detach()
        coming_max = torch.max(t, dim=0)[0].float().cpu()
        coming_min = torch.min(t, dim=0)[0].float().cpu()
        midpoint = (coming_max + coming_min) / 2.0
        if name in act_shifts:
            act_shifts[name] = 0.99 * act_shifts[name] + 0.01 * midpoint
        else:
            act_shifts[name] = midpoint

    def hook_fn(_m: nn.Module, x: Any, _y: Any, name: str) -> None:
        if isinstance(x, tuple):
            x = x[0]
        if not torch.is_tensor(x):
            return
        stat_scale(name, x)
        stat_shift(name, x)

    import functools

    hooks: List[Any] = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(functools.partial(hook_fn, name=name)))

    n = min(int(num_samples), len(dataloader))
    for i in range(n):
        inp = dataloader[i][0]
        model(inp.to(device, non_blocking=True))

    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass

    return act_scales, act_shifts


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
@torch.no_grad()
def eval_on_datasets(
    model: nn.Module,
    tok,
    device: str,
    dataset_specs: List[str],
    cfg: EvalConfig,
    default_split: str,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {"datasets": {}}

    for spec in dataset_specs:
        ds, split = parse_eval_spec(spec, default_split=default_split)
        ds_key = f"{ds[0]}:{ds[1]}" if ds[1] is not None else ds[0]
        key = f"{ds_key}::{split}"

        loader, nblocks, ntok = build_eval_loader(
            tok,
            hf_tuple(ds),
            cfg.block_size,
            cfg.batch_size_eval,
            split=split,
            max_fraction=cfg.eval_fraction,
        )
        ppl, nll, _ = perplexity(model, loader, device)

        results["datasets"][key] = {
            "dataset": ds_key,
            "split": split,
            "block_size": int(cfg.block_size),
            "batch_size_eval": int(cfg.batch_size_eval),
            "eval_fraction": float(cfg.eval_fraction),
            "nblocks": int(nblocks),
            "tokens": int(ntok),
            "nll": float(nll),
            "ppl": float(ppl),
        }

        print(f"[EVAL] {key}  blocks={nblocks} tokens={ntok}  NLL={nll:.6f}  PPL={ppl:.4f}")

    return results


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate PPL on multiple datasets with optional PTQ: fake-quant, GPTQ, GPTQ+fake activation quant, SmoothQuant, SpinQuant, or OmniQuant."
    )

    p.add_argument("--model-dir", type=str, required=True, help="Local checkpoint dir OR HF repo id.")
    p.add_argument("--local-files-only", action="store_true", help="Do not download if HF id; use local cache only.")

    p.add_argument("--datasets", type=str, nargs="+", default=list(DEFAULT_EVAL_DATASETS))
    p.add_argument("--default-split", type=str, default="validation")

    p.add_argument("--device", type=str, default=("cuda:0" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", type=str, default="auto", help="auto|fp32|fp16|bf16")

    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--batch-size-eval", type=int, default=1)
    p.add_argument("--eval-fraction", type=float, default=0.01)

    p.add_argument(
        "--ptq",
        type=str,
        default="fakequant",
        choices=["none", "fakequant", "gptq", "gptq_fakeact", "smoothquant", "spinquant", "omniquant"],
        help="PTQ method: none | fakequant | gptq | gptq_fakeact | smoothquant | spinquant | omniquant",
    )

    # Fake activation quant config (fakequant and gptq_fakeact)
    p.add_argument("--act-quant", type=str, default="per_token", choices=["per_token", "per_tensor"])
    p.add_argument("--act-bits", type=int, default=4)
    p.add_argument("--act-location", type=str, default="input", choices=["input", "output", "both"])

    # Fake weight quant config (fakequant)
    p.add_argument("--weight-quant", type=str, default="per_channel", choices=["per_channel", "per_tensor"])
    p.add_argument("--weight-bits", type=int, default=4)
    p.add_argument("--quantize-bmm-input", action="store_true", default=False)

    # GPTQ config (gptq and gptq_fakeact) - default calibration moved to WikiText-2 train
    p.add_argument("--gptq-calib", type=str, default="wikitext:wikitext-2-raw-v1::train")
    p.add_argument("--gptq-nsamples", type=int, default=1024)
    p.add_argument("--gptq-seqlen", type=int, default=256)
    p.add_argument("--gptq-wbits", type=int, default=4)
    p.add_argument("--gptq-groupsize", type=int, default=-1)
    p.add_argument("--gptq-percdamp", type=float, default=0.01)
    p.add_argument("--gptq-act-order", action="store_true")
    p.add_argument("--gptq-true-sequential", action="store_true")
    p.add_argument("--gptq-static-groups", action="store_true")
    p.add_argument("--gptq-sym", action="store_true")

    p.add_argument(
        "--quant-reload-instead-of-deepcopy",
        action="store_true",
        help="If set, reload fresh model for PTQ rather than deepcopying the dense model.",
    )

    # SmoothQuant config (defaults already WikiText-2; set split=train for calibration)
    p.add_argument(
        "--smoothquant-dir",
        type=Path,
        default=None,
        help="Path to sft-smoothquant directory (the one that contains smoothquant/). If omitted, inferred as sibling of wiki-qwen3-sft.",
    )
    p.add_argument("--smoothquant-alpha", type=float, default=0.5, help="SmoothQuant alpha (single value).")
    p.add_argument("--smoothquant-calib-num-samples", type=int, default=1024)
    p.add_argument("--smoothquant-calib-seq-len", type=int, default=256)
    p.add_argument("--smoothquant-calib-dataset-name", type=str, default="wikitext")
    p.add_argument("--smoothquant-calib-dataset-subset", type=str, default="wikitext-2-raw-v1")
    p.add_argument("--smoothquant-calib-dataset-split", type=str, default="train")
    p.add_argument("--smoothquant-device-map", type=str, default="cuda:0")
    p.add_argument(
        "--smoothquant-act-scales-path",
        type=Path,
        default=None,
        help="If provided, skip act-scale generation and load scales from this file (torch.load).",
    )

    # SpinQuant config
    p.add_argument(
        "--spinquant-dir",
        type=Path,
        default=None,
        help="Path to SpinQuant-main (optional). If omitted, we try to infer it near /workspace.",
    )

    p.add_argument("--spinquant-run-optimize-rotations", action="store_true")
    p.add_argument("--spinquant-rot-dir", type=Path, default=None)
    p.add_argument("--spinquant-optimized-rotation-path", type=Path, default=None)
    p.add_argument("--spinquant-optimize-extra-args", type=str, nargs="*", default=[])

    p.add_argument("--spinquant-w-bits", type=int, default=4)
    p.add_argument("--spinquant-a-bits", type=int, default=4)
    p.add_argument("--spinquant-model-max-length", type=int, default=256)
    p.add_argument("--spinquant-access-token", type=str, default=None)
    p.add_argument("--spinquant-ptq-kv", type=str, nargs="*", default=[])
    p.add_argument("--spinquant-force-bf16", action="store_true")
    p.add_argument("--spinquant-force-fp16", action="store_true")

    # OmniQuant config (default calibration moved to wikitext2)
    p.add_argument(
        "--omniquant-dir",
        type=Path,
        default=None,
        help="Path to OmniQuant-main. If omitted, inferred as /workspace/OmniQuant-main.",
    )
    p.add_argument(
        "--omniquant-net",
        type=str,
        default=None,
        help="OmniQuant --net value (e.g., opt-1.3b, llama-7b). If omitted, we guess from HF config.",
    )
    p.add_argument(
        "--omniquant-cache-dir",
        type=Path,
        default=None,
        help="OmniQuant cache_dir (defaults to out_dir/cache_omniquant).",
    )
    p.add_argument(
        "--omniquant-save-quantized",
        action="store_true",
        help="If set, save the OmniQuant-quantized model under out_dir/omniquant_ckpt.",
    )
    p.add_argument(
        "--omniquant-calib-dataset",
        type=str,
        default="wikitext2",
        choices=["wikitext103", "wikitext2", "ptb", "c4", "mix", "pile"],
    )
    p.add_argument("--omniquant-nsamples", type=int, default=1024)
    p.add_argument("--omniquant-seqlen", type=int, default=256)
    p.add_argument("--omniquant-batch-size", type=int, default=8)
    p.add_argument("--omniquant-alpha", type=float, default=0.5)
    p.add_argument("--omniquant-wbits", type=int, default=4)
    p.add_argument("--omniquant-abits", type=int, default=4)
    p.add_argument("--omniquant-group-size", type=int, default=0, help="Group size (0 means None).")
    p.add_argument("--omniquant-let-lr", type=float, default=1e-5)
    p.add_argument("--omniquant-lwc-lr", type=float, default=1e-5)
    p.add_argument("--omniquant-wd", type=float, default=0.0)
    p.add_argument("--omniquant-epochs", type=int, default=10)
    p.add_argument("--omniquant-let", action="store_true")
    p.add_argument("--omniquant-lwc", action="store_true")
    p.add_argument("--omniquant-aug-loss", action="store_true")
    p.add_argument("--omniquant-symmetric", action="store_true")
    p.add_argument("--omniquant-disable-zero-point", action="store_true")
    p.add_argument("--omniquant-a-dynamic-method", type=str, default="per_token", choices=["per_token"])
    p.add_argument("--omniquant-w-dynamic-method", type=str, default="per_channel", choices=["per_channel"])
    p.add_argument("--omniquant-limit", type=int, default=-1)
    p.add_argument("--omniquant-multigpu", action="store_true")
    p.add_argument("--omniquant-deactive-amp", action="store_true")
    p.add_argument(
        "--omniquant-attn-implementation",
        type=str,
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
    )
    p.add_argument("--omniquant-act-scales", type=Path, default=None)
    p.add_argument("--omniquant-act-shifts", type=Path, default=None)
    p.add_argument(
        "--omniquant-generate-missing-act-scales-shifts",
        action="store_true",
        help=(
            "If --omniquant-let and act_scales/shifts files are missing, regenerate them from the calibration loader "
            "and save to --omniquant-act-scales/--omniquant-act-shifts (or defaults)."
        ),
    )
    p.add_argument(
        "--omniquant-eval-quant-before-ft",
        action="store_true",
        help="If set, run eval_on_datasets before omniquant finetuning as well.",
    )
    p.add_argument("--omniquant-real-quant", action="store_true", help="Pass through to OmniQuant args.")
    p.add_argument("--omniquant-resume", type=str, default=None)

    p.add_argument("--out-dir", type=Path, default=None)
    return p


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------
def _load_dense_model(model_dir: Path, dtype: Optional[torch.dtype], device: str) -> nn.Module:
    _ = AutoConfig.from_pretrained(str(model_dir))

    last_err: Optional[Exception] = None
    for trust_remote_code in (False, True):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                str(model_dir),
                torch_dtype=dtype if dtype is not None else "auto",
                device_map=None,
                trust_remote_code=trust_remote_code,
            )
            return model.to(device).eval()
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Failed to load model from {model_dir}. Last error: {last_err}") from last_err


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    args = build_parser().parse_args()
    set_seed(int(args.seed))

    model_dir = _resolve_model_dir(args.model_dir, local_files_only=bool(args.local_files_only))
    device = str(args.device)
    dtype = _resolve_dtype(args.dtype)

    out_dir = (args.out_dir or (model_dir.parent / f"eval_{datetime.now().strftime('%Y%m%d-%H%M%S')}")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = EvalConfig(
        block_size=int(args.block_size),
        batch_size_eval=int(args.batch_size_eval),
        eval_fraction=_validate_fraction(float(args.eval_fraction)),
    )

    print(f"[INFO] resolved_model_dir={model_dir}")
    print(f"[INFO] device={device} dtype={args.dtype}")
    print(
        f"[INFO] loader: block_size={cfg.block_size} batch_size_eval={cfg.batch_size_eval} "
        f"eval_fraction={cfg.eval_fraction}"
    )
    print(f"[INFO] ptq={args.ptq}")

    tok = _make_tokenizer(str(model_dir))
    t0 = time.time()

    # Always compute dense baseline first
    model_dense = _load_dense_model(model_dir, dtype=dtype, device=device)

    if not _is_supported_causallm(model_dense):
        print(
            f"[WARN] Unrecognized model family (type={type(model_dense)} model_type={_model_type(model_dense)}). Continuing."
        )

    dense_results = eval_on_datasets(
        model_dense,
        tok,
        device=device,
        dataset_specs=list(args.datasets),
        cfg=cfg,
        default_split=str(args.default_split),
    )

    ptq_results: Optional[Dict[str, Any]] = None
    ptq_meta: Dict[str, Any] = {"ptq": str(args.ptq)}

    if args.ptq == "none":
        _release_cuda_models(model_dense)

    elif args.ptq == "fakequant":
        model_q = (
            _load_dense_model(model_dir, dtype=dtype, device=device)
            if args.quant_reload_instead_of_deepcopy
            else copy.deepcopy(model_dense).to(device).eval()
        )
        if args.quant_reload_instead_of_deepcopy:
            _release_cuda_models(model_dense)

        fake_quantize_model(
            model_q,
            weight_quant=str(args.weight_quant),
            act_quant=str(args.act_quant),
            quantize_bmm_input=bool(args.quantize_bmm_input),
            weight_bits=int(args.weight_bits),
            act_bits=int(args.act_bits),
            act_location=str(args.act_location),
            exclude_lm_head=True,
            use_lwc=False,
        )

        ptq_meta["fakequant"] = {
            "weight_quant": str(args.weight_quant),
            "weight_bits": int(args.weight_bits),
            "act_quant": str(args.act_quant),
            "act_bits": int(args.act_bits),
            "act_location": str(args.act_location),
            "quantize_bmm_input": bool(args.quantize_bmm_input),
        }

        ptq_results = eval_on_datasets(
            model_q,
            tok,
            device=device,
            dataset_specs=list(args.datasets),
            cfg=cfg,
            default_split=str(args.default_split),
        )

        _release_cuda_models(model_q, model_dense)

    elif args.ptq in {"gptq", "gptq_fakeact"}:
        if not _looks_llama_like_decoder(model_dense):
            _release_cuda_models(model_dense)
            raise RuntimeError("GPTQ path expects model.model.layers (LLaMA/Mistral/Mixtral-style decoder).")

        model_q = (
            _load_dense_model(model_dir, dtype=dtype, device="cpu")
            if args.quant_reload_instead_of_deepcopy
            else copy.deepcopy(model_dense).to("cpu").eval()
        )
        if args.quant_reload_instead_of_deepcopy:
            _release_cuda_models(model_dense)

        meta = gptq_quantize_weights_inplace_llama_like(
            model=model_q,
            tok=tok,
            device=device,
            calib_spec=str(args.gptq_calib),
            default_split=str(args.default_split),
            nsamples=int(args.gptq_nsamples),
            seqlen=int(args.gptq_seqlen),
            wbits=int(args.gptq_wbits),
            groupsize=int(args.gptq_groupsize),
            percdamp=float(args.gptq_percdamp),
            act_order=bool(args.gptq_act_order),
            true_sequential=bool(args.gptq_true_sequential),
            static_groups=bool(args.gptq_static_groups),
            sym=bool(args.gptq_sym),
        )

        model_q = model_q.to(device).eval()

        hooks: List[Any] = []
        if args.ptq == "gptq_fakeact":
            hooks = add_activation_quant_hooks(
                model_q,
                act_quant=str(args.act_quant),
                act_bits=int(args.act_bits),
                act_location=str(args.act_location),
                exclude_lm_head=True,
            )

        ptq_meta["gptq"] = meta
        if args.ptq == "gptq_fakeact":
            ptq_meta["fakeact"] = {
                "act_quant": str(args.act_quant),
                "act_bits": int(args.act_bits),
                "act_location": str(args.act_location),
            }

        ptq_results = eval_on_datasets(
            model_q,
            tok,
            device=device,
            dataset_specs=list(args.datasets),
            cfg=cfg,
            default_split=str(args.default_split),
        )

        _remove_hooks(hooks)
        _release_cuda_models(model_q, model_dense)

    elif args.ptq == "smoothquant":
        sq_root = (args.smoothquant_dir.resolve() if args.smoothquant_dir is not None else _default_smoothquant_root())
        smooth_lm, _sq_quantize_model, sq_root, sq_pkg_dir = _smoothquant_import(sq_root)

        if args.smoothquant_act_scales_path is not None:
            scales_path = args.smoothquant_act_scales_path.expanduser().resolve()
            if not scales_path.exists():
                raise FileNotFoundError(f"--smoothquant-act-scales-path not found: {scales_path}")
        else:
            scales_path = (out_dir / "smoothquant_act_scales.pt").resolve()
            calib_jsonl = (out_dir / "smoothquant_calib.jsonl").resolve()

            _run_smoothquant_generate_act_scales(
                sq_root=sq_root,
                sq_pkg_dir=sq_pkg_dir,
                model_name_or_dir=str(model_dir),
                out_scales_path=scales_path,
                dataset_path=calib_jsonl,
                dataset_name=str(args.smoothquant_calib_dataset_name),
                dataset_subset=str(args.smoothquant_calib_dataset_subset)
                if args.smoothquant_calib_dataset_subset
                else None,
                dataset_split=str(args.smoothquant_calib_dataset_split),
                num_samples=int(args.smoothquant_calib_num_samples),
                seq_len=int(args.smoothquant_calib_seq_len),
                device_map=str(args.smoothquant_device_map),
                extra_env=None,
            )

        act_scales = torch.load(scales_path, map_location="cpu")
        print(f"[SQ] act_scales loaded: {scales_path}")

        model_q = (
            _load_dense_model(model_dir, dtype=dtype, device=device)
            if args.quant_reload_instead_of_deepcopy
            else copy.deepcopy(model_dense).to(device).eval()
        )
        if args.quant_reload_instead_of_deepcopy:
            _release_cuda_models(model_dense)

        act_scales_dev: Dict[str, Any] = {}
        for k, v in act_scales.items():
            act_scales_dev[k] = v.to(device) if isinstance(v, torch.Tensor) else v

        alpha = float(args.smoothquant_alpha)
        print(f"[SQ] Applying smooth_lm(alpha={alpha})")
        smooth_lm(model_q, act_scales_dev, alpha)

        print("[SQ] Applying tools/fake_quant.quantize_model(...)")
        fake_quantize_model(
            model_q,
            weight_quant=str(args.weight_quant),
            act_quant=str(args.act_quant),
            quantize_bmm_input=bool(args.quantize_bmm_input),
            weight_bits=int(args.weight_bits),
            act_bits=int(args.act_bits),
            act_location=str(args.act_location),
            exclude_lm_head=True,
            use_lwc=False,
        )

        ptq_meta["smoothquant"] = {
            "smoothquant_root": str(sq_root),
            "smoothquant_pkg_dir": str(sq_pkg_dir),
            "alpha": float(alpha),
            "act_scales_path": str(scales_path),
            "calib": {
                "num_samples": int(args.smoothquant_calib_num_samples),
                "seq_len": int(args.smoothquant_calib_seq_len),
                "dataset_name": str(args.smoothquant_calib_dataset_name),
                "dataset_subset": str(args.smoothquant_calib_dataset_subset),
                "dataset_split": str(args.smoothquant_calib_dataset_split),
                "device_map": str(args.smoothquant_device_map),
            },
            "quant": {
                "weight_quant": str(args.weight_quant),
                "weight_bits": int(args.weight_bits),
                "act_quant": str(args.act_quant),
                "act_bits": int(args.act_bits),
                "act_location": str(args.act_location),
                "quantize_bmm_input": bool(args.quantize_bmm_input),
                "note": "Applied tools/fake_quant.py fake quant after smooth_lm; sq_quantize_model not used.",
            },
        }

        ptq_results = eval_on_datasets(
            model_q,
            tok,
            device=device,
            dataset_specs=list(args.datasets),
            cfg=cfg,
            default_split=str(args.default_split),
        )

        _release_cuda_models(model_q, model_dense)

    elif args.ptq == "spinquant":
        spinq_dir = args.spinquant_dir.resolve() if args.spinquant_dir is not None else _find_spinquant_dir()
        if spinq_dir is None or not spinq_dir.exists():
            raise FileNotFoundError("Could not locate SpinQuant-main. Provide --spinquant-dir /path/to/SpinQuant-main.")
        spinq_dir = spinq_dir.resolve()
        print(f"[SPINQ] spinquant_dir={spinq_dir}")

        spin_rot_dir = (args.spinquant_rot_dir or (out_dir / "spinquant_rotations")).resolve()

        sq_dtype = _pick_spinquant_torch_dtype(args, dtype)
        bf16_flag = (sq_dtype == torch.bfloat16)
        fp16_flag = (sq_dtype == torch.float16)

        input_model = str(args.model_dir)
        access_token = args.spinquant_access_token

        if bool(args.spinquant_run_optimize_rotations):
            _run_spinquant_optimize_rotations(
                spinq_dir=spinq_dir,
                model_id_or_dir=str(args.model_dir),
                rot_out_dir=spin_rot_dir,
                extra_args=list(args.spinquant_optimize_extra_args),
                log_path=out_dir / "logs" / "spinquant_optimize_rotations.log",
            )

        if args.spinquant_optimized_rotation_path is not None:
            rotation_path = args.spinquant_optimized_rotation_path.expanduser().resolve()
            if not rotation_path.exists():
                raise FileNotFoundError(f"--spinquant-optimized-rotation-path not found: {rotation_path}")
        else:
            rotation_path = _find_rotation_file(spin_rot_dir)

        if rotation_path is None or not rotation_path.exists():
            raise FileNotFoundError(
                "Could not find SpinQuant rotation file.\n"
                f"Looked in: {spin_rot_dir}\n"
                "Fix by either:\n"
                "  - setting --spinquant-rot-dir to SpinQuant's actual --output_rotation_path directory, or\n"
                "  - passing --spinquant-optimized-rotation-path to the exact file."
            )
        print(f"[SPINQ] rotation_path={rotation_path}")

        spinquant_ptq_model, SpinQuantLlamaForCausalLM = _spinquant_import(spinq_dir)

        import transformers as _spinq_transformers  # local import to reduce collisions

        config = _spinq_transformers.AutoConfig.from_pretrained(input_model, token=access_token)

        process_word_embeddings = False
        if getattr(config, "tie_word_embeddings", False):
            config.tie_word_embeddings = False
            process_word_embeddings = True

        if bool(args.quant_reload_instead_of_deepcopy):
            _release_cuda_models(model_dense)

        model_q = SpinQuantLlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=input_model,
            config=config,
            torch_dtype=sq_dtype,
            token=access_token,
        )
        if process_word_embeddings:
            model_q.lm_head.weight.data = model_q.model.embed_tokens.weight.data.clone()

        model_q = model_q.to(device).eval()
        model_q.config.use_cache = False

        spin_out_dir = out_dir / "spinquant_internal"
        spin_logs_dir = spin_out_dir / "logs"
        spin_out_dir.mkdir(parents=True, exist_ok=True)
        spin_logs_dir.mkdir(parents=True, exist_ok=True)

        extra_kv = _parse_kv_args(list(args.spinquant_ptq_kv))

        model_args, training_args, ptq_args = _spinquant_build_args_via_process_args_ptq(
            spinq_dir=spinq_dir,
            input_model=input_model,
            optimized_rotation_path=rotation_path,
            seed=int(args.seed),
            w_bits=int(args.spinquant_w_bits),
            a_bits=int(args.spinquant_a_bits),
            model_max_length=int(args.spinquant_model_max_length),
            per_device_eval_batch_size=1,
            bf16=bool(bf16_flag),
            fp16=bool(fp16_flag),
            rotate=True,
            rotate_mode=str(extra_kv.get("rotate_mode", "hadamard")),
            fp32_had=bool(extra_kv.get("fp32_had", False)),
            w_rtn=bool(extra_kv.get("w_rtn", False)),
            a_groupsize=int(extra_kv["a_groupsize"]) if "a_groupsize" in extra_kv else None,
            w_groupsize=int(extra_kv["w_groupsize"]) if "w_groupsize" in extra_kv else None,
            percdamp=float(extra_kv["percdamp"]) if "percdamp" in extra_kv else None,
            nsamples=int(extra_kv["nsamples"]) if "nsamples" in extra_kv else None,
            act_order=bool(extra_kv["act_order"]) if "act_order" in extra_kv else None,
            load_qmodel_path=str(extra_kv["load_qmodel_path"]) if "load_qmodel_path" in extra_kv else None,
            save_qmodel_path=str(extra_kv["save_qmodel_path"]) if "save_qmodel_path" in extra_kv else None,
            export_to_et=bool(extra_kv["export_to_et"]) if "export_to_et" in extra_kv else None,
            output_dir=spin_out_dir,
            logging_dir=spin_logs_dir,
        )

        try:
            model_args.access_token = access_token
        except Exception:
            pass

        model_q = spinquant_ptq_model(ptq_args, model_q, model_args)
        model_q = model_q.to(device).eval()
        model_q.config.use_cache = False

        ptq_meta["spinquant"] = {
            "spinquant_dir": str(spinq_dir),
            "rot_dir": str(spin_rot_dir),
            "rotation_path": str(rotation_path),
            "w_bits": int(args.spinquant_w_bits),
            "a_bits": int(args.spinquant_a_bits),
            "model_max_length": int(args.spinquant_model_max_length),
            "torch_dtype": str(sq_dtype),
            "ran_optimize_rotations": bool(args.spinquant_run_optimize_rotations),
            "optimize_extra_args": list(args.spinquant_optimize_extra_args),
            "ptq_extra_kv": extra_kv,
            "input_model": input_model,
            "used_access_token": bool(access_token),
        }

        ptq_results = eval_on_datasets(
            model_q,
            tok,
            device=device,
            dataset_specs=list(args.datasets),
            cfg=cfg,
            default_split=str(args.default_split),
        )

        _release_cuda_models(model_q, model_dense)

    elif args.ptq == "omniquant":
        omni_dir = (args.omniquant_dir.resolve() if args.omniquant_dir is not None else OMNIQUANT_DIR.resolve())
        if not omni_dir.exists():
            raise FileNotFoundError(
                f"OmniQuant dir not found: {omni_dir}. Provide --omniquant-dir /path/to/OmniQuant-main."
            )

        LMClass, omniquant_fn, get_loaders, map_layers_to_multi_gpus, get_lowest_occupied_gpu, omni_utils = _omniquant_import(
            omni_dir
        )

        oq_run_dir = (out_dir / "omniquant_run").resolve()
        oq_run_dir.mkdir(parents=True, exist_ok=True)
        oq_cache_dir = (
            args.omniquant_cache_dir.resolve()
            if args.omniquant_cache_dir is not None
            else (out_dir / "cache_omniquant").resolve()
        )
        oq_cache_dir.mkdir(parents=True, exist_ok=True)

        logger = omni_utils.create_logger(oq_run_dir)

        class _OQArgs:
            pass

        oq = _OQArgs()
        oq.model = str(model_dir)
        oq.cache_dir = str(oq_cache_dir)
        oq.output_dir = str(oq_run_dir)
        oq.save_dir = str((out_dir / "omniquant_ckpt").resolve()) if bool(args.omniquant_save_quantized) else None
        oq.resume = args.omniquant_resume
        oq.real_quant = bool(args.omniquant_real_quant)

        oq.calib_dataset = str(args.omniquant_calib_dataset)
        oq.nsamples = int(args.omniquant_nsamples)
        oq.batch_size = int(args.omniquant_batch_size)
        oq.seed = int(args.seed)

        oq.tasks = ""
        oq.eval_ppl = False  # We evaluate it here instead
        oq.num_fewshot = 0
        oq.wbits = int(args.omniquant_wbits)
        oq.abits = int(args.omniquant_abits)

        oq.group_size = None if int(args.omniquant_group_size) <= 0 else int(args.omniquant_group_size)
        oq.alpha = float(args.omniquant_alpha)
        oq.let_lr = float(args.omniquant_let_lr)
        oq.lwc_lr = float(args.omniquant_lwc_lr)
        oq.wd = float(args.omniquant_wd)
        oq.epochs = int(args.omniquant_epochs)

        oq.let = bool(args.omniquant_let)
        oq.lwc = bool(args.omniquant_lwc)
        oq.aug_loss = bool(args.omniquant_aug_loss)
        oq.symmetric = bool(args.omniquant_symmetric)
        oq.disable_zero_point = bool(args.omniquant_disable_zero_point)
        oq.a_dynamic_method = str(args.omniquant_a_dynamic_method)
        oq.w_dynamic_method = str(args.omniquant_w_dynamic_method)
        oq.limit = int(args.omniquant_limit)

        oq.multigpu = bool(args.omniquant_multigpu)
        oq.deactive_amp = bool(args.omniquant_deactive_amp) or _omniquant_should_deactivate_amp(oq.wbits, oq.abits)
        oq.attn_implementation = str(args.omniquant_attn_implementation)

        dense_cfg = AutoConfig.from_pretrained(str(model_dir))
        fallback_net = str(args.omniquant_net) if args.omniquant_net is not None else str(Path(args.model_dir).name)
        oq.net = (
            str(args.omniquant_net)
            if args.omniquant_net is not None
            else _infer_omniquant_net_from_config(dense_cfg, fallback=fallback_net)
        )
        oq.model_family = (
            oq.net.split("-")[0]
            if "-" in oq.net
            else (getattr(dense_cfg, "model_type", "model") or "model")
        )

        if args.omniquant_act_scales is not None:
            oq.act_scales = str(args.omniquant_act_scales.expanduser().resolve())
        else:
            oq.act_scales = str((omni_dir / "act_scales" / f"{oq.net}.pt").resolve())

        if args.omniquant_act_shifts is not None:
            oq.act_shifts = str(args.omniquant_act_shifts.expanduser().resolve())
        else:
            oq.act_shifts = str((omni_dir / "act_shifts" / f"{oq.net}.pt").resolve())

        random.seed(oq.seed)
        torch.manual_seed(oq.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(oq.seed)

        torch.backends.cudnn.benchmark = True
        lm = LMClass(oq)

        target = torch.device(device)
        try:
            lm.device = target
        except Exception:
            pass
        try:
            lm._device = str(target)
        except Exception:
            pass

        lm.seqlen = int(args.omniquant_seqlen)
        try:
            lm.model.eval()
        except Exception:
            pass

        try:
            for param in lm.model.parameters():
                param.requires_grad = False
        except Exception:
            pass

        oq.weight_quant_params = {
            "n_bits": oq.wbits,
            "per_channel_axes": [0],
            "symmetric": oq.symmetric,
            "dynamic_method": oq.w_dynamic_method,
            "group_size": oq.group_size,
            "lwc": oq.lwc,
            "disable_zero_point": oq.disable_zero_point,
        }
        oq.act_quant_params = {
            "n_bits": oq.abits,
            "per_channel_axes": [],
            "symmetric": True,
            "dynamic_method": oq.a_dynamic_method,
            "disable_zero_point": oq.disable_zero_point,
        }
        oq.q_quant_params = {
            "n_bits": oq.abits,
            "per_channel_axes": [],
            "symmetric": True,
            "dynamic_method": oq.a_dynamic_method,
        }
        oq.k_quant_params = {
            "n_bits": oq.abits,
            "per_channel_axes": [],
            "symmetric": True,
            "dynamic_method": oq.a_dynamic_method,
        }
        oq.v_quant_params = {
            "n_bits": oq.abits,
            "per_channel_axes": [],
            "symmetric": True,
            "dynamic_method": oq.a_dynamic_method,
        }
        oq.p_quant_params = {
            "n_bits": 16,
            "metric": "fix0to1",
        }

        if oq.multigpu:
            try:
                gpu_id = get_lowest_occupied_gpu(wait_memory=5000)
                lm._device = f"cuda:{gpu_id}"
                logger.info(f"[OQ] set quantization in gpu {gpu_id}")
            except Exception as e:
                logger.info(f"[OQ] multigpu requested but GPU mapping failed: {e}")

        calib_meta: Dict[str, Any] = {
            "calib_dataset": oq.calib_dataset,
            "nsamples": oq.nsamples,
            "seqlen": lm.seqlen,
            "seed": oq.seed,
        }

        # ---- Calibration loader selection (default now wikitext2) ----
        if oq.calib_dataset in {"wikitext103", "wikitext2"}:
            calib_tag = f"{oq.calib_dataset}_train"
            cpath = _cache_path_for_omni_calib(
                cache_dir=Path(oq.cache_dir),
                model_family=str(oq.model_family),
                model_id=str(oq.model),
                calib_tag=calib_tag,
                seqlen=int(lm.seqlen),
                nsamples=int(oq.nsamples),
                seed=int(oq.seed),
            )

            if cpath.exists():
                dataloader = torch.load(cpath, map_location="cpu")
                logger.info(f"[OQ] load calibration from {cpath}")
                calib_meta["cache_path"] = str(cpath)
                calib_meta["cache_hit"] = True
            else:
                if oq.calib_dataset == "wikitext103":
                    dataloader = _build_wikitext103_calib_loader(
                        str(args.model_dir), int(lm.seqlen), int(oq.nsamples), int(oq.seed)
                    )
                else:
                    dataloader = _build_wikitext2_calib_loader(
                        str(args.model_dir), int(lm.seqlen), int(oq.nsamples), int(oq.seed)
                    )
                torch.save(dataloader, cpath)
                logger.info(f"[OQ] saved calibration to {cpath}")
                calib_meta["cache_path"] = str(cpath)
                calib_meta["cache_hit"] = False
        else:
            cache_dataloader = (
                Path(oq.cache_dir) / f"dataloader_{oq.model_family}_{oq.calib_dataset}_{oq.nsamples}.cache"
            )
            if cache_dataloader.exists():
                dataloader = torch.load(cache_dataloader, map_location="cpu")
                logger.info(f"[OQ] load calibration from {cache_dataloader}")
                calib_meta["cache_path"] = str(cache_dataloader)
                calib_meta["cache_hit"] = True
            else:
                dataloader, _ = get_loaders(
                    oq.calib_dataset,
                    nsamples=oq.nsamples,
                    seed=oq.seed,
                    model=oq.model,
                    seqlen=lm.seqlen,
                )
                torch.save(dataloader, cache_dataloader)
                calib_meta["cache_path"] = str(cache_dataloader)
                calib_meta["cache_hit"] = False

        oq_pre_results: Optional[Dict[str, Any]] = None
        if bool(args.omniquant_eval_quant_before_ft):
            try:
                lm.model = lm.model.to(device).eval()
            except Exception:
                pass
            oq_pre_results = eval_on_datasets(
                lm.model,
                tok,
                device=device,
                dataset_specs=list(args.datasets),
                cfg=cfg,
                default_split=str(args.default_split),
            )

        act_scales = None
        act_shifts = None
        if bool(oq.let):
            scales_path = Path(oq.act_scales)
            shifts_path = Path(oq.act_shifts)

            # Keep behavior consistent with your prior script: regenerate (or overwrite) in-process.
            logger.info(
                f"[OQ] act_scales/shifts missing; regenerating into:\n  scales={scales_path}\n  shifts={shifts_path}"
            )
            scales_path.parent.mkdir(parents=True, exist_ok=True)
            shifts_path.parent.mkdir(parents=True, exist_ok=True)

            lm.model = lm.model.to(device).eval()
            print("[OQ] lm.model param device:", next(lm.model.parameters()).device, flush=True)

            gen_scales, gen_shifts = _omniquant_generate_act_scales_shifts(
                model=lm.model,
                dataloader=dataloader,
                num_samples=int(oq.nsamples),
            )
            torch.save(gen_scales, str(scales_path))
            torch.save(gen_shifts, str(shifts_path))
            logger.info("[OQ] regeneration done and saved.")

            if not scales_path.exists():
                raise FileNotFoundError(f"[OQ] --let set but act_scales not found: {scales_path}")
            if not shifts_path.exists():
                raise FileNotFoundError(f"[OQ] --let set but act_shifts not found: {shifts_path}")
            act_scales = torch.load(str(scales_path), map_location="cpu")
            act_shifts = torch.load(str(shifts_path), map_location="cpu")

        logger.info("[OQ] === start quantization ===")
        tick = time.time()

        print("[OQ] device arg:", device, flush=True)
        print("[OQ] lm._device:", getattr(lm, "_device", None), flush=True)
        print("[OQ] model param device:", next(lm.model.parameters()).device, flush=True)
        print("[OQ] model dtype:", next(lm.model.parameters()).dtype, flush=True)

        omniquant_fn(lm, oq, dataloader, act_scales, act_shifts, logger)
        logger.info(f"[OQ] quantization done in {time.time() - tick:.2f}s")

        # ---- transformers/OmniQuant kwarg compatibility (apply immediately) ----
        try:
            lm.model.config.use_cache = False
        except Exception:
            pass
        compat1 = _patch_omniquant_decoder_layers_hf_compat(lm.model)
        print(f"[OQ] patched decoder layers for HF kwargs: {compat1}", flush=True)

        if oq.save_dir is not None:
            Path(oq.save_dir).mkdir(parents=True, exist_ok=True)
            try:
                lm.model.save_pretrained(oq.save_dir)
                lm.tokenizer.save_pretrained(oq.save_dir)
                logger.info(f"[OQ] saved quantized model to: {oq.save_dir}")
            except Exception as e:
                logger.info(f"[OQ] warning: failed to save_pretrained: {e}")

        # Evaluate with the same eval pipeline as other methods
        try:
            lm.model = lm.model.to(device).eval()
            lm.model.config.use_cache = False
        except Exception:
            pass

        # Patch again after move/save (some forks rewrap layers on device transfer)
        compat2 = _patch_omniquant_decoder_layers_hf_compat(lm.model)
        print(f"[OQ] patched-after-move decoder layers for HF kwargs: {compat2}", flush=True)

        # Patch HF LlamaRMSNorm to unwrap tuple hidden_states
        n_rms = _patch_hf_llama_rmsnorm_unwrap()
        print(f"[OQ] patched HF LlamaRMSNorm unwrap: {n_rms}", flush=True)
        
        # Patch HF Qwen3 norm to unwrap tuple hidden_states
        n_qwen_norm = _patch_hf_qwen3_norm_unwrap()
        print(f"[OQ] patched HF Qwen3 norm unwrap: {n_qwen_norm}", flush=True)

        ptq_results = eval_on_datasets(
            lm.model,
            tok,
            device=device,
            dataset_specs=list(args.datasets),
            cfg=cfg,
            default_split=str(args.default_split),
        )

        ptq_meta["omniquant"] = {
            "omniquant_dir": str(omni_dir),
            "run_dir": str(oq_run_dir),
            "cache_dir": str(oq_cache_dir),
            "net": str(oq.net),
            "model_family": str(oq.model_family),
            "device": str(device),
            "seqlen": int(lm.seqlen),
            "calib": calib_meta,
            "wbits": int(oq.wbits),
            "abits": int(oq.abits),
            "group_size": (None if oq.group_size is None else int(oq.group_size)),
            "alpha": float(oq.alpha),
            "let": bool(oq.let),
            "lwc": bool(oq.lwc),
            "let_lr": float(oq.let_lr),
            "lwc_lr": float(oq.lwc_lr),
            "wd": float(oq.wd),
            "epochs": int(oq.epochs),
            "aug_loss": bool(oq.aug_loss),
            "symmetric": bool(oq.symmetric),
            "disable_zero_point": bool(oq.disable_zero_point),
            "a_dynamic_method": str(oq.a_dynamic_method),
            "w_dynamic_method": str(oq.w_dynamic_method),
            "multigpu": bool(oq.multigpu),
            "deactive_amp": bool(oq.deactive_amp),
            "attn_implementation": str(oq.attn_implementation),
            "act_scales": str(oq.act_scales),
            "act_shifts": str(oq.act_shifts),
            "saved_ckpt_dir": str(oq.save_dir) if oq.save_dir is not None else None,
            "eval_before_ft": bool(args.omniquant_eval_quant_before_ft),
            "pre_eval_results": oq_pre_results,
        }

        _release_cuda_models(lm, model_dense)

    else:
        raise ValueError(f"Unknown --ptq {args.ptq}")

    elapsed = time.time() - t0

    final: Dict[str, Any] = {
        "input_model_dir_or_id": str(args.model_dir),
        "resolved_model_dir": str(model_dir),
        "device": device,
        "dtype": str(args.dtype),
        "loader": {
            "block_size": int(cfg.block_size),
            "batch_size_eval": int(cfg.batch_size_eval),
            "eval_fraction": float(cfg.eval_fraction),
            "default_split": str(args.default_split),
        },
        "dense": dense_results,
        "ptq": ptq_meta,
        "ptq_results": ptq_results,
        "elapsed_sec": float(elapsed),
        "timestamp": datetime.now().isoformat(),
        "paths": {
            "tools_dir": str(TOOLS_DIR),
            "gptq_dir": str(GPTQ_DIR),
            "omniquant_dir": str(OMNIQUANT_DIR),
            "project_dir": str(PROJECT_DIR),
        },
    }

    out_path = out_dir / "eval_results.json"
    out_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(f"[SAVE] {out_path}")
    print("[DONE]")


if __name__ == "__main__":
    main()
