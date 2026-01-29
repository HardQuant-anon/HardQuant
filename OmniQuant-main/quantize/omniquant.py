#!/usr/bin/env python3
# /workspace/OmniQuant-main/quantize/omniquant.py
#
# Qwen3 integration note:
# HF Qwen3Attention.forward requires `position_embeddings=(cos, sin)` (and often `cache_position`)
# to be passed in. When we call decoder layers directly (as OmniQuant does), we must provide these.
#
# This file:
#  - Captures `position_embeddings` and `cache_position` from the first-layer call during the Catcher pass.
#  - For Qwen3, uses an HF-forward-preserving wrapper around the HF decoder layer:
#       * keeps HF forward intact (so attention semantics are identical)
#       * replaces nn.Linear with QuantLinear
#       * implements LET/LWC smoothing hooks compatible with the existing transformation utilities
#  - Keeps Llama/OPT/Falcon/Mixtral behavior as before.

from __future__ import annotations

import copy
import gc
import math
import os
import pdb
from contextlib import nullcontext
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

import utils
from models.int_falcon_layer import QuantFalconDecoderLayer
from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_opt_layer import QuantOPTDecoderLayer
from quantize.int_linear import QuantLinear
from quantize.utils import (
    clear_temp_variable,
    get_omni_parameters,
    let_parameters,
    lwc_parameters,
    omni_state_dict,
    register_scales_and_zeros,
    set_quant_state,
    smooth_and_quant_inplace,
    smooth_and_quant_temporary,
)

from quantize.omni_norm import OmniLlamaRMSNorm

# smoothing helpers (same ones used by the llama quant layers)
from models.transformation import (  # noqa: F401
    truncate_number,
    smooth_ln_fcs_temporary,
    smooth_ln_fcs_inplace,
    smooth_fc_fc_temporary,
    smooth_fc_fc_inplace,
    smooth_q_k_temporary,
    smooth_q_k_inplace,
)

try:
    import auto_gptq.nn_modules.qlinear.qlinear_cuda as qlinear_cuda
    import auto_gptq.nn_modules.qlinear.qlinear_triton as qlinear_triton
except Exception:
    print("auto_gptq is required for real quantization")


def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, QuantLinear)}


def add_new_module(name, original_module, added_module):
    levels = name.split(".")
    if len(levels) > 1:
        mod_ = original_module
        for l_idx in range(len(levels) - 1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], added_module)
    else:
        setattr(original_module, name, added_module)


def _hs(out):
    # unwrap tuple/list hidden_states without ever slicing a Tensor's batch dim
    if isinstance(out, (tuple, list)):
        out = out[0]
    if not torch.is_tensor(out):
        raise RuntimeError(f"[OQ] unexpected layer output type: {type(out)}")
    return out


# -----------------------------------------------------------------------------
# Qwen3 fix: ensure every decoder layer has attention_type (HF Qwen3 requires it)
# -----------------------------------------------------------------------------
def _get_attention_type_for_layer(model, layer, layer_idx: int) -> str:
    if hasattr(layer, "attention_type"):
        at = getattr(layer, "attention_type")
        if isinstance(at, str) and at:
            return at

    lt = getattr(getattr(model, "config", None), "layer_types", None)
    if isinstance(lt, (list, tuple)) and 0 <= int(layer_idx) < len(lt):
        if isinstance(lt[layer_idx], str) and lt[layer_idx]:
            return lt[layer_idx]

    return "full_attention"


def _ensure_attention_type_on_layers(model, layers) -> int:
    fixed = 0
    for i, lyr in enumerate(layers):
        if not hasattr(lyr, "attention_type"):
            setattr(lyr, "attention_type", _get_attention_type_for_layer(model, lyr, i))
            fixed += 1
    return fixed


def _replace_linears_inplace(root: nn.Module, weight_qparams: dict, act_qparams: dict, skip_gate: bool = False):
    """
    Replace nn.Linear modules with QuantLinear in-place, preserving outer-module forward.
    """
    for name, module in list(root.named_modules()):
        if isinstance(module, nn.Linear):
            if skip_gate and ("gate" in name):
                continue
            qlin = QuantLinear(module, weight_qparams, act_qparams)
            add_new_module(name, root, qlin)


def _find_attr(obj: Any, names: Tuple[str, ...]):
    for n in names:
        if hasattr(obj, n) and getattr(obj, n) is not None:
            return getattr(obj, n)
    return None


class QuantHFQwen3DecoderLayer(nn.Module):
    """
    HF-forward-preserving wrapper for a Qwen3 decoder layer.

    Critical: wrap RMSNorm modules (layer norms + q_norm/k_norm) with OmniLlamaRMSNorm
    so that `smooth_ln_fcs_temporary/inplace` can affect the forward via temp_weight/temp_bias.
    """

    def __init__(self, hf_layer: nn.Module, args, dev: torch.device, dtype: torch.dtype):
        super().__init__()
        self.hf = hf_layer
        self.let = False
        self.use_weight_quant = False
        self.use_act_quant = False
        self._dtype = dtype

        # Find expected submodules
        self.self_attn = _find_attr(self.hf, ("self_attn", "attn", "attention"))
        self.mlp = _find_attr(self.hf, ("mlp", "feed_forward", "ffn"))

        if self.self_attn is None or self.mlp is None:
            raise RuntimeError("Could not locate attention/MLP on HF Qwen3 decoder layer.")

        # ---- Replace linears inside the HF layer so HF forward uses QuantLinear ----
        _replace_linears_inplace(self.hf, args.weight_quant_params, args.act_quant_params, skip_gate=False)

        # ---- Wrap decoder RMSNorms so temp_* is honored ----
        # Qwen3DecoderLayer typically uses input_layernorm/post_attention_layernorm
        in_ln = _find_attr(self.hf, ("input_layernorm", "input_layer_norm", "ln_1", "norm1"))
        post_ln = _find_attr(self.hf, ("post_attention_layernorm", "post_attention_layer_norm", "ln_2", "norm2"))

        if in_ln is None or post_ln is None:
            raise RuntimeError("Could not locate expected layernorm modules on HF Qwen3 decoder layer.")

        def _wrap_rmsnorm(mod):
            # Qwen3RMSNorm uses variance_epsilon; fall back safely
            eps = getattr(mod, "variance_epsilon", None)
            if eps is None:
                eps = getattr(mod, "eps", 1e-6)
            return OmniLlamaRMSNorm(mod, eps=eps)

        # Replace on the HF layer so HF forward uses the wrapper
        # (this is the important part)
        self.input_layernorm = _wrap_rmsnorm(in_ln)
        self.post_attention_layernorm = _wrap_rmsnorm(post_ln)

        # Set back into hf layer attributes
        if hasattr(self.hf, "input_layernorm"):
            self.hf.input_layernorm = self.input_layernorm
        elif hasattr(self.hf, "input_layer_norm"):
            self.hf.input_layer_norm = self.input_layernorm
        elif hasattr(self.hf, "ln_1"):
            self.hf.ln_1 = self.input_layernorm
        elif hasattr(self.hf, "norm1"):
            self.hf.norm1 = self.input_layernorm

        if hasattr(self.hf, "post_attention_layernorm"):
            self.hf.post_attention_layernorm = self.post_attention_layernorm
        elif hasattr(self.hf, "post_attention_layer_norm"):
            self.hf.post_attention_layer_norm = self.post_attention_layernorm
        elif hasattr(self.hf, "ln_2"):
            self.hf.ln_2 = self.post_attention_layernorm
        elif hasattr(self.hf, "norm2"):
            self.hf.norm2 = self.post_attention_layernorm

        # ---- Wrap attention q_norm/k_norm (Qwen3Attention has these) ----
        # HF Qwen3Attention uses q_norm/k_norm in its forward; these must also honor temp_*.
        if hasattr(self.self_attn, "q_norm") and isinstance(self.self_attn.q_norm, nn.Module):
            self.self_attn.q_norm = _wrap_rmsnorm(self.self_attn.q_norm)
        if hasattr(self.self_attn, "k_norm") and isinstance(self.self_attn.k_norm, nn.Module):
            self.self_attn.k_norm = _wrap_rmsnorm(self.self_attn.k_norm)

        # ---- Ensure expected projections exist and are QuantLinear ----
        for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            if not hasattr(self.self_attn, proj_name):
                raise RuntimeError(f"HF Qwen3Attention missing {proj_name}.")
            if not isinstance(getattr(self.self_attn, proj_name), QuantLinear):
                raise RuntimeError(f"HF Qwen3Attention.{proj_name} was not replaced by QuantLinear.")

        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            if not hasattr(self.mlp, proj_name):
                raise RuntimeError(f"HF Qwen3 MLP missing {proj_name}.")
            if not isinstance(getattr(self.mlp, proj_name), QuantLinear):
                raise RuntimeError(f"HF Qwen3 MLP.{proj_name} was not replaced by QuantLinear.")

        # Keep attention_type if present
        if hasattr(self.hf, "attention_type"):
            self.attention_type = getattr(self.hf, "attention_type")

        self.to(dev)
        self.hf.to(dev)

    def forward(self, *args, **kwargs):
        return self.hf(*args, **kwargs)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, QuantLinear):
                m.set_quant_state(weight_quant, act_quant)

    # ---- Parameter selectors expected by quantize/utils.py ----
    def let_parameters(self, use_shift=True):
        params = []
        template = "smooth" if use_shift else "smooth_scale"
        for n, p in self.named_parameters():
            if template in n:
                params.append(p)
        return iter(params)

    def lwc_parameters(self):
        params = []
        for n, p in self.named_parameters():
            if "bound_factor" in n:
                params.append(p)
        return iter(params)

    def omni_parameters(self, use_shift=True):
        params = []
        template = "smooth" if use_shift else "smooth_scale"
        for n, p in self.named_parameters():
            if ("bound_factor" in n) or (template in n):
                params.append(p)
        return iter(params)

    def omni_state_dict(self, destination=None, prefix="", keep_vars=False):
        if destination is None:
            destination = {}
        for name, param in self.named_parameters():
            if ("smooth" in name) or ("bound_factor" in name):
                destination[prefix + name] = param if keep_vars else param.detach()
        return destination

    # ---- LET/LWC helpers ----
    def smooth_and_quant_temporary(self):
        # Delegate to the global util which uses model.input_layernorm etc.
        # This wrapper now ensures those norms honor temp_* in forward.
        smooth_and_quant_temporary(self, args=self._oq_args, isllama=True)  # will be overridden below

    def clear_temp_variable(self):
        clear_temp_variable(self)

    @torch.no_grad()
    def smooth_and_quant_inplace(self):
        smooth_and_quant_inplace(self, args=self._oq_args, isllama=True)  # will be overridden below

    def register_scales_and_zeros(self):
        register_scales_and_zeros(self)


def omniquant(lm, args, dataloader, act_scales, act_shifts, logger=None):
    logger.info("Starting ...")

    model = lm.model

    dev = lm.device
    if isinstance(dev, str):
        dev = torch.device(dev)

    use_cache = model.config.use_cache
    model.config.use_cache = False

    net_l = (getattr(args, "net", "") or "").lower()
    model_type = (getattr(model.config, "model_type", "") or "").lower()

    def _is_qwen():
        return ("qwen" in net_l) or ("qwen" in model_type) or (model_type in {"qwen2", "qwen2_moe", "qwen3"})

    def _is_qwen3():
        return ("qwen3" in net_l) or (model_type == "qwen3")

    def _is_llama_like():
        return hasattr(model, "model") and hasattr(model.model, "layers")

    is_llama = False
    is_qwen3_hf = False

    # ---------------------------
    # Choose layer list / prefixes
    # ---------------------------
    if "llama" in net_l or model_type == "llama":
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        DecoderLayer = QuantLlamaDecoderLayer
        pairs = {"q_proj": "qkv", "o_proj": "out", "up_proj": "fc1"}
        layer_name_prefix = "model.layers"

    elif _is_qwen() and _is_llama_like():
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        pairs = {"q_proj": "qkv", "o_proj": "out", "up_proj": "fc1"}
        layer_name_prefix = "model.layers"

        fixed0 = _ensure_attention_type_on_layers(model, layers)
        logger.info(f"[OQ][QWEN] ensured attention_type on base layers: fixed={fixed0}")

        if _is_qwen3():
            is_qwen3_hf = True
        else:
            # Qwen2 fallback: use custom layer if you still have it
            from models.int_qwen_layer import QuantQwenDecoderLayer as DecoderLayer  # type: ignore
            DecoderLayer = DecoderLayer

    elif "mixtral" in net_l or model_type == "mixtral":
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        layer_name_prefix = "model.layers"

    elif "opt" in net_l or model_type == "opt":
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
        DecoderLayer = QuantOPTDecoderLayer
        pairs = {"q_proj": "qkv", "out_proj": "out", "fc1": "fc1"}
        layer_name_prefix = "model.decoder.layers"

    elif "falcon" in net_l or model_type == "falcon":
        layers = model.transformer.h
        model.transformer.word_embeddings.to(dev)
        model.transformer.ln_f.to(dev)
        model.lm_head.to(dev)
        DecoderLayer = QuantFalconDecoderLayer
        layer_name_prefix = "model.transformer.h"

    else:
        raise ValueError(
            f"Only support opt/llama/falcon/mixtral/qwen-like now. "
            f"(args.net={getattr(args,'net',None)}, model_type={model_type})"
        )

    layers[0] = layers[0].to(dev)

    # AMP selection
    if args.deactive_amp and args.epochs > 0:
        dtype = torch.float
        traincast = nullcontext
    else:
        dtype = torch.float16
        traincast = lambda: torch.amp.autocast(device_type="cuda", dtype=torch.float16)

    inps = torch.zeros((args.nsamples, lm.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache: Dict[str, Any] = {"i": 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False
            if hasattr(module, "attention_type"):
                self.attention_type = module.attention_type

        def __getattr__(self, name):
            if name not in {"module", "is_llama", "attention_type"} and "module" in self.__dict__:
                mod = self.__dict__["module"]
                if hasattr(mod, name):
                    return getattr(mod, name)
            return super().__getattr__(name)

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1

            am = kwargs.get("attention_mask", None)
            if torch.is_tensor(am) and am.device != dev:
                am = am.to(dev)
            cache["attention_mask"] = am

            pid = kwargs.get("position_ids", None)
            if torch.is_tensor(pid) and pid.device != dev:
                pid = pid.to(dev)
            cache["position_ids"] = pid

            # Qwen3: capture position_embeddings=(cos,sin) and cache_position
            pe = kwargs.get("position_embeddings", None)
            if isinstance(pe, (tuple, list)) and len(pe) == 2 and torch.is_tensor(pe[0]) and torch.is_tensor(pe[1]):
                cos, sin = pe
                if cos.device != dev:
                    cos = cos.to(dev)
                if sin.device != dev:
                    sin = sin.to(dev)
                cache["position_embeddings"] = (cos, sin)
            else:
                cache["position_embeddings"] = None

            cp = kwargs.get("cache_position", None)
            if torch.is_tensor(cp) and cp.device != dev:
                cp = cp.to(dev)
            cache["cache_position"] = cp

            raise ValueError

    layers[0] = Catcher(layers[0])
    layers[0].is_llama = is_llama

    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass

    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()

    # move embeddings/norm back to CPU
    if is_llama or (_is_qwen() and _is_llama_like()):
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    elif "opt" in net_l:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif "falcon" in net_l:
        model.transformer.word_embeddings = model.transformer.word_embeddings.cpu()
    else:
        raise ValueError("Only support opt/llama/falcon/mixtral/qwen-like now")

    torch.cuda.empty_cache()

    quant_inps = inps
    fp_inps = copy.deepcopy(inps)
    fp_inps_2 = copy.deepcopy(inps) if args.aug_loss else None

    attention_mask = cache.get("attention_mask", None)

    if torch.is_tensor(attention_mask):
        attention_mask_batch = (
            attention_mask.repeat(args.batch_size, 1, 1, 1)
            if args.deactive_amp
            else attention_mask.repeat(args.batch_size, 1, 1, 1).float()
        )
    else:
        logger.info("No attention mask caught from the first layer. Assuming attention works without a mask.")
        attention_mask_batch = None

    position_ids = cache.get("position_ids", None)
    if not torch.is_tensor(position_ids):
        position_ids = None

    # Qwen3: captured from the real model forward
    position_embeddings = cache.get("position_embeddings", None)
    cache_position = cache.get("cache_position", None)

    position_embeddings_batch = None
    cache_position_batch = None

    if is_qwen3_hf:
        if position_embeddings is None:
            raise RuntimeError(
                "[OQ][QWEN3] position_embeddings was not captured in Catcher. "
                "This means the first-layer forward did not receive it from the model."
            )
        cos, sin = position_embeddings
        if cos.shape[0] == 1 and args.batch_size > 1:
            cos_b = cos.expand(args.batch_size, *cos.shape[1:]).contiguous()
            sin_b = sin.expand(args.batch_size, *sin.shape[1:]).contiguous()
        else:
            cos_b, sin_b = cos, sin
        position_embeddings_batch = (cos_b, sin_b)

        if torch.is_tensor(cache_position):
            if cache_position.dim() == 1 and args.batch_size > 1:
                cache_position_batch = cache_position.unsqueeze(0).expand(args.batch_size, -1).contiguous()
            elif cache_position.dim() == 2 and cache_position.shape[0] == 1 and args.batch_size > 1:
                cache_position_batch = cache_position.expand(args.batch_size, -1).contiguous()
            else:
                cache_position_batch = cache_position
        else:
            cp = torch.arange(lm.seqlen, device=dev, dtype=torch.long)
            cache_position_batch = cp.unsqueeze(0).expand(args.batch_size, -1).contiguous() if args.batch_size > 1 else cp

    loss_func = torch.nn.MSELoss()

    if args.resume:
        omni_parameters = torch.load(args.resume)
    else:
        omni_parameters = {}

    # ---------------------------------------------------------
    # Quantize layer by layer
    # ---------------------------------------------------------
    for i in range(len(layers)):
        logger.info(f"=== Start quantize layer {i} ===")
        layer = layers[i].to(dev)

        # Build qlayer
        if is_qwen3_hf:
            if not hasattr(layer, "attention_type"):
                layer.attention_type = _get_attention_type_for_layer(model, layer, i)
            qlayer = QuantHFQwen3DecoderLayer(hf_layer=copy.deepcopy(layer), args=args, dev=dev, dtype=dtype)
            qlayer.let = bool(args.let)
            if hasattr(layer, "attention_type"):
                qlayer.attention_type = getattr(layer, "attention_type")

        elif "mixtral" in net_l:
            qlayer = copy.deepcopy(layer)
            for name, module in qlayer.named_modules():
                if isinstance(module, torch.nn.Linear) and "gate" not in name:
                    quantlinear = QuantLinear(module, args.weight_quant_params, args.act_quant_params)
                    add_new_module(name, qlayer, quantlinear)

        else:
            qlayer = DecoderLayer(lm.model.config, layer, args)

        qlayer = qlayer.to(dev)

        # Teacher targets from full precision: compute using quant disabled
        set_quant_state(qlayer, weight_quant=False, act_quant=False)

        if args.epochs > 0:
            with torch.no_grad():
                with traincast():
                    for j in range(args.nsamples):
                        if is_qwen3_hf:
                            fp_inps[j] = _hs(
                                qlayer(
                                    fp_inps[j].unsqueeze(0),
                                    attention_mask=attention_mask,
                                    position_embeddings=position_embeddings,
                                    cache_position=cache_position,
                                )
                            )
                            if args.aug_loss:
                                fp_inps_2[j] = _hs(
                                    qlayer(
                                        quant_inps[j].unsqueeze(0),
                                        attention_mask=attention_mask,
                                        position_embeddings=position_embeddings,
                                        cache_position=cache_position,
                                    )
                                )
                        else:
                            fp_inps[j] = _hs(
                                qlayer(
                                    fp_inps[j].unsqueeze(0),
                                    attention_mask=attention_mask,
                                    position_ids=position_ids,
                                )
                            )
                            if args.aug_loss:
                                fp_inps_2[j] = _hs(
                                    qlayer(
                                        quant_inps[j].unsqueeze(0),
                                        attention_mask=attention_mask,
                                        position_ids=position_ids,
                                    )
                                )

        # Init smooth params and enable activation quant (student)
        set_quant_state(qlayer, weight_quant=False, act_quant=True)
        qlayer.let = bool(args.let)

        use_shift = True
        if is_llama or args.abits == 16:
            use_shift = False

        # Register LET parameters
        if args.let:
            q_out = qlayer.self_attn.q_proj.out_features
            qlayer.register_parameter(
                "qkt_smooth_scale",
                torch.nn.Parameter(torch.ones(q_out, device=dev, dtype=dtype)),
            )

            for name, module in qlayer.named_modules():
                if not isinstance(module, QuantLinear):
                    continue

                # Qwen3 wrapper includes "hf." prefix; act_scales keys typically do not
                name_key = name[3:] if name.startswith("hf.") else name

                for key in pairs.keys():
                    if key in name_key:
                        k = f"{layer_name_prefix}.{i}.{name_key}"
                        act = act_scales[k].to(device=dev, dtype=dtype).clamp(min=1e-5)
                        weight = module.weight.abs().max(dim=0)[0].clamp(min=1e-5)
                        scale = (act.pow(args.alpha) / weight.pow(1 - args.alpha)).clamp(min=1e-5)
                        if use_shift and not is_llama:
                            shift = act_shifts[k].to(device=dev, dtype=dtype)
                        else:
                            shift = torch.zeros_like(scale)
                        qlayer.register_parameter(f"{pairs[key]}_smooth_shift", torch.nn.Parameter(shift))
                        qlayer.register_parameter(f"{pairs[key]}_smooth_scale", torch.nn.Parameter(scale))

        if args.resume:
            qlayer.load_state_dict(omni_parameters[i], strict=False)

        # Finetune LET/LWC params
        if args.epochs > 0:
            with torch.no_grad():
                qlayer.float()

            optimizer = torch.optim.AdamW(
                [
                    {"params": let_parameters(qlayer, use_shift), "lr": args.let_lr},
                    {"params": lwc_parameters(qlayer), "lr": args.lwc_lr},
                ],
                weight_decay=args.wd,
            )
            loss_scaler = utils.NativeScalerWithGradNormCount()

            for epochs in range(args.epochs):
                loss_list = []
                norm_list = []

                steps = args.nsamples // args.batch_size
                if steps <= 0:
                    raise RuntimeError(f"nsamples={args.nsamples} batch_size={args.batch_size} gives zero steps")

                for j in range(steps):
                    index = j * args.batch_size

                    with traincast():
                        smooth_and_quant_temporary(qlayer, args, is_llama)

                        if is_qwen3_hf:
                            quant_out = _hs(
                                qlayer(
                                    quant_inps[index : index + args.batch_size],
                                    attention_mask=attention_mask_batch,
                                    position_embeddings=position_embeddings_batch,
                                    cache_position=cache_position_batch,
                                )
                            )
                        else:
                            quant_out = _hs(
                                qlayer(
                                    quant_inps[index : index + args.batch_size],
                                    attention_mask=attention_mask_batch,
                                    position_ids=position_ids,
                                )
                            )

                        targ = fp_inps[index : index + args.batch_size]
                        pred = quant_out

                        if targ.dim() == 2:
                            targ = targ.unsqueeze(0)
                        if pred.dim() == 2:
                            pred = pred.unsqueeze(0)
                        if targ.shape != pred.shape:
                            raise RuntimeError(
                                f"[OQ] recon mismatch layer={i} epoch={epochs} targ={tuple(targ.shape)} pred={tuple(pred.shape)}"
                            )

                        loss = loss_func(targ, pred)
                        if args.aug_loss:
                            targ2 = fp_inps_2[index : index + args.batch_size]
                            if targ2.dim() == 2:
                                targ2 = targ2.unsqueeze(0)
                            if targ2.shape != pred.shape:
                                raise RuntimeError(
                                    f"[OQ] recon mismatch (aug) layer={i} epoch={epochs} targ2={tuple(targ2.shape)} pred={tuple(pred.shape)}"
                                )
                            loss = loss + loss_func(targ2, pred)

                    if not math.isfinite(loss.item()):
                        logger.info("Loss is NAN/INF, stopping training")
                        pdb.set_trace()

                    loss_list.append(loss.detach().cpu())
                    optimizer.zero_grad()
                    norm = loss_scaler(loss, optimizer, parameters=get_omni_parameters(qlayer, use_shift)).cpu()
                    norm_list.append(norm.data)

                loss_mean = torch.stack(loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                logger.info(
                    f"layer {i} iter {epochs} loss:{loss_mean} norm:{norm_mean} "
                    f"max memory_allocated {torch.cuda.max_memory_allocated(lm._device) / 1024**2} "
                )

            clear_temp_variable(qlayer)
            del optimizer

        qlayer.half()

        # Apply final smoothing + weight quant in-place
        smooth_and_quant_inplace(qlayer, args, is_llama)

        # Propagate quant inputs to next layer
        if args.epochs > 0:
            with torch.no_grad():
                with traincast():
                    for j in range(args.nsamples):
                        if is_qwen3_hf:
                            quant_inps[j] = _hs(
                                qlayer(
                                    quant_inps[j].unsqueeze(0),
                                    attention_mask=attention_mask,
                                    position_embeddings=position_embeddings,
                                    cache_position=cache_position,
                                )
                            )
                        else:
                            quant_inps[j] = _hs(
                                qlayer(
                                    quant_inps[j].unsqueeze(0),
                                    attention_mask=attention_mask,
                                    position_ids=position_ids,
                                )
                            )

            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")

            if _is_qwen() and _is_llama_like() and not hasattr(layers[i], "attention_type"):
                layers[i].attention_type = _get_attention_type_for_layer(model, layer, i)

            omni_parameters[i] = omni_state_dict(qlayer)
            torch.save(omni_parameters, os.path.join(args.output_dir, "omni_parameters.pth"))
        else:
            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")

            if _is_qwen() and _is_llama_like() and not hasattr(layers[i], "attention_type"):
                layers[i].attention_type = _get_attention_type_for_layer(model, layer, i)

        # Optional: real quant packing
        if args.real_quant:
            assert args.wbits in [2, 3, 4] and args.abits >= 16
            named_linears = get_named_linears(qlayer)
            for name, module in named_linears.items():
                scales = module.weight_quantizer.scales
                zeros = module.weight_quantizer.zeros
                group_size = module.weight_quantizer.group_size
                dim0 = module.weight.shape[0]
                scales = scales.view(dim0, -1)
                zeros = zeros.view(dim0, -1)
                if args.wbits == 3:
                    q_linear = qlinear_cuda.QuantLinear(
                        args.wbits,
                        group_size,
                        module.in_features,
                        module.out_features,
                        not module.bias is None,
                    )
                else:
                    q_linear = qlinear_triton.QuantLinear(
                        args.wbits,
                        group_size,
                        module.in_features,
                        module.out_features,
                        not module.bias is None,
                    )
                q_linear.pack(module.cpu(), scales.float().cpu(), zeros.float().cpu())
                add_new_module(name, qlayer, q_linear)
                print(f"pack quantized {name} finished")
                del module

        del layer
        torch.cuda.empty_cache()

    del inps
    del quant_inps
    del fp_inps
    del fp_inps_2
    torch.cuda.empty_cache()
    gc.collect()

    model.config.use_cache = use_cache

    # Final defensive pass: ensure all Qwen layers still have attention_type
    if _is_qwen() and _is_llama_like():
        fixed_end = _ensure_attention_type_on_layers(model, model.model.layers)
        logger.info(f"[OQ][QWEN] ensured attention_type on final layers: fixed={fixed_end}")

    return model
