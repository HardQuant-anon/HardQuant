import torch
from torch import nn
from functools import partial
from typing import Callable, Iterator, List, Optional, Set, Tuple


def _unwrap_wrapped_model(model: nn.Module) -> nn.Module:
    """
    Unwrap common wrappers (FSDP/DDP) for robust module discovery.
    """
    cur: nn.Module = model
    # unwrap FSDP if available
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        while isinstance(cur, FSDP):
            cur = cur.module  # type: ignore[assignment]
    except Exception:
        pass
    # unwrap DDP
    if isinstance(cur, torch.nn.parallel.DistributedDataParallel):
        cur = cur.module
    return cur


def get_output_embedding_module(model: nn.Module) -> Optional[nn.Module]:
    """
    Robustly get the output-embedding / lm_head module if present.
    """
    m = _unwrap_wrapped_model(model)
    out_emb = None
    try:
        out_emb = m.get_output_embeddings()
    except Exception:
        out_emb = None
    if out_emb is None:
        out_emb = getattr(m, "lm_head", None)
    return out_emb if isinstance(out_emb, nn.Module) else None


def iter_linear_weight_modules(
    model: nn.Module,
    exclude_lm_head: bool = True,
) -> Iterator[Tuple[str, nn.Linear]]:
    """
    Yield nn.Linear modules whose weights should be quantized (deduped).
    """
    m = _unwrap_wrapped_model(model)

    out_emb = get_output_embedding_module(m)
    exclude_ids: Set[int] = set()
    if exclude_lm_head and out_emb is not None:
        exclude_ids.add(id(out_emb))

    seen: Set[int] = set()
    for name, module in m.named_modules():
        if exclude_lm_head and (name.split(".")[-1] == "lm_head" or id(module) in exclude_ids):
            continue

        if isinstance(module, nn.Linear):
            mid = id(module)
            if mid not in seen:
                seen.add(mid)
                yield name, module

        base = getattr(module, "base_layer", None)
        if isinstance(base, nn.Linear) and base is not module:
            bid = id(base)
            if bid not in seen:
                seen.add(bid)
                yield f"{name}.base_layer", base


def iter_activation_quant_modules(
    model: nn.Module,
    exclude_lm_head: bool = True,
) -> Iterator[Tuple[str, nn.Module]]:
    """
    Yield modules whose forward activations are quantized (hooks attach here).
    """
    m = _unwrap_wrapped_model(model)

    out_emb = get_output_embedding_module(m)
    exclude_ids: Set[int] = set()
    if exclude_lm_head and out_emb is not None:
        exclude_ids.add(id(out_emb))

    seen: Set[int] = set()
    for name, module in m.named_modules():
        if exclude_lm_head and (name.split(".")[-1] == "lm_head" or id(module) in exclude_ids):
            continue

        take = False
        if isinstance(module, nn.Linear):
            take = True
        base = getattr(module, "base_layer", None)
        if isinstance(base, nn.Linear) and base is not module:
            take = True

        if take:
            mid = id(module)
            if mid not in seen:
                seen.add(mid)
                yield name, module


@torch.no_grad()
def quantize_weight_per_channel_absmax(w: torch.Tensor, n_bits: int) -> torch.Tensor:
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_weight_per_tensor_absmax(w: torch.Tensor, n_bits: int) -> torch.Tensor:
    scales = w.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_activation_per_token_absmax(t: torch.Tensor, n_bits: int):
    if not isinstance(t, torch.Tensor):
        return t
    tq = t.clone()
    orig_shape = tq.shape
    tq_flat = tq.view(-1, orig_shape[-1])
    scales = tq_flat.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    tq_flat.div_(scales).round_().mul_(scales)
    return tq_flat.view(orig_shape)


@torch.no_grad()
def quantize_activation_per_tensor_absmax(t: torch.Tensor, n_bits: int):
    if not isinstance(t, torch.Tensor):
        return t
    # IMPORTANT: do NOT modify `t` in-place
    tq = t.clone()
    scales = tq.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    tq.div_(scales).round_().mul_(scales)
    return tq


@torch.no_grad()
def maybe_clip_weights_with_lwc(linear: nn.Linear, w: torch.Tensor) -> torch.Tensor:
    alpha = getattr(linear, "lwc_alpha", None)
    if alpha is None:
        return w

    eps = 1e-8
    a = alpha.data
    if a.numel() == 1:
        a = a.clamp(min=eps, max=1.0)
        clip = w.abs().max().clamp(min=eps) * a
        return w.clamp(-clip, clip)

    a = a.clamp(min=eps, max=1.0)
    if a.dim() != 2 or a.size(1) != 1 or a.size(0) != w.size(0):
        a = a.view(w.size(0), 1).clamp(min=eps, max=1.0)

    amax = w.abs().max(dim=-1, keepdim=True)[0].clamp(min=eps)
    clip = (amax * a).clamp(min=eps)
    return w.clamp(-clip, clip)


@torch.no_grad()
def quantize_linear_weight_in_place(linear: nn.Linear, weight_quant: str, n_bits: int) -> None:
    w = linear.weight.data
    w = maybe_clip_weights_with_lwc(linear, w)

    if weight_quant == "per_channel":
        linear.weight.data = quantize_weight_per_channel_absmax(w, n_bits=n_bits)
    elif weight_quant == "per_tensor":
        linear.weight.data = quantize_weight_per_tensor_absmax(w, n_bits=n_bits)
    else:
        raise ValueError(f"Invalid weight_quant: {weight_quant}")


@torch.no_grad()
def quantize_all_linear_weights(
    model: nn.Module,
    weight_quant: str = "per_channel",
    weight_bits: int = 8,
    exclude_lm_head: bool = True,
) -> None:
    for _name, lin in iter_linear_weight_modules(model, exclude_lm_head=exclude_lm_head):
        quantize_linear_weight_in_place(lin, weight_quant=weight_quant, n_bits=weight_bits)


def make_activation_quant_fn(act_quant: str, act_bits: int) -> Callable[[torch.Tensor], torch.Tensor]:
    act_quant = (act_quant or "per_token").lower()
    if act_quant == "per_token":
        return partial(quantize_activation_per_token_absmax, n_bits=act_bits)
    if act_quant == "per_tensor":
        return partial(quantize_activation_per_tensor_absmax, n_bits=act_bits)
    raise ValueError(f"Invalid act_quant: {act_quant}")


def apply_act_fn_to_out(act_fn: Callable[[torch.Tensor], torch.Tensor], out):
    if isinstance(out, torch.Tensor):
        return act_fn(out)
    if isinstance(out, (list, tuple)):
        out_list = [act_fn(t) if isinstance(t, torch.Tensor) else t for t in out]
        return type(out)(out_list)
    return out


def add_activation_quant_hooks(
    model: nn.Module,
    act_quant: str = "per_token",
    act_bits: int = 8,
    act_location: str = "output",
    exclude_lm_head: bool = True,
) -> List[torch.utils.hooks.RemovableHandle]:
    if act_location not in {"input", "output", "both"}:
        raise ValueError(f"Invalid act_location: {act_location}. Must be 'input', 'output', or 'both'.")

    m = _unwrap_wrapped_model(model)

    out_emb = get_output_embedding_module(m)
    exclude_ids: Set[int] = set()
    if exclude_lm_head and out_emb is not None:
        exclude_ids.add(id(out_emb))

    act_fn = make_activation_quant_fn(act_quant, act_bits)
    hooks: List[torch.utils.hooks.RemovableHandle] = []

    def fwd_hook(_module: nn.Module, _inp, out):
        return apply_act_fn_to_out(act_fn, out)

    def pre_hook(_module: nn.Module, inp):
        if not inp:
            return inp
        x0 = inp[0]
        if isinstance(x0, torch.Tensor):
            x0q = act_fn(x0)
            return (x0q,) + tuple(inp[1:])
        return inp

    # Build wrapper->base_layer relationships so we can:
    # - quantize INPUT at base_layer (true Linear input)
    # - quantize OUTPUT at wrapper (includes LoRA additions, etc.)
    wrappers: List[Tuple[str, nn.Module, nn.Linear]] = []
    base_layer_ids: Set[int] = set()

    for name, module in m.named_modules():
        if exclude_lm_head and (name.split(".")[-1] == "lm_head" or id(module) in exclude_ids):
            continue
        base = getattr(module, "base_layer", None)
        if isinstance(base, nn.Linear) and base is not module:
            if exclude_lm_head and id(base) in exclude_ids:
                continue
            wrappers.append((name, module, base))
            base_layer_ids.add(id(base))

    # Decide hook targets with dedupe
    seen_in: Set[int] = set()
    seen_out: Set[int] = set()

    # INPUT targets
    if act_location in {"input", "both"}:
        # Prefer base_layer inputs for wrappers
        for _name, _wrap, base in wrappers:
            bid = id(base)
            if bid not in seen_in:
                seen_in.add(bid)
                hooks.append(base.register_forward_pre_hook(pre_hook))

        # Plain nn.Linear modules not used as base_layer of a wrapper
        for name, module in m.named_modules():
            if exclude_lm_head and (name.split(".")[-1] == "lm_head" or id(module) in exclude_ids):
                continue
            if isinstance(module, nn.Linear):
                mid = id(module)
                if mid in base_layer_ids:
                    continue
                if mid not in seen_in:
                    seen_in.add(mid)
                    hooks.append(module.register_forward_pre_hook(pre_hook))

    # OUTPUT targets
    if act_location in {"output", "both"}:
        # Prefer wrapper outputs for wrappers (captures LoRA additions etc.)
        for _name, wrap, _base in wrappers:
            wid = id(wrap)
            if wid not in seen_out:
                seen_out.add(wid)
                hooks.append(wrap.register_forward_hook(fwd_hook))

        # Plain nn.Linear modules not used as base_layer of a wrapper
        for name, module in m.named_modules():
            if exclude_lm_head and (name.split(".")[-1] == "lm_head" or id(module) in exclude_ids):
                continue
            if isinstance(module, nn.Linear):
                mid = id(module)
                if mid in base_layer_ids:
                    continue
                if mid not in seen_out:
                    seen_out.add(mid)
                    hooks.append(module.register_forward_hook(fwd_hook))

    return hooks



def quantize_model(
    model: nn.Module,
    weight_quant: str = "per_channel",
    act_quant: str = "per_token",
    quantize_bmm_input: bool = False,
    weight_bits: int = 8,
    act_bits: int = 8,
    act_location: str = "input",
    exclude_lm_head: bool = True,
):
    _ = quantize_bmm_input
    quantize_all_linear_weights(
        model,
        weight_quant=weight_quant,
        weight_bits=weight_bits,
        exclude_lm_head=exclude_lm_head,
    )
    add_activation_quant_hooks(
        model,
        act_quant=act_quant,
        act_bits=act_bits,
        act_location=act_location,
        exclude_lm_head=exclude_lm_head,
    )
    return model

