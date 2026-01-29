# tools/fake_quant.py
import torch
from torch import nn
from torch.nn import functional as F
from functools import partial
from typing import Callable, Dict, Iterator, List, Optional, Set, Tuple


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _unwrap_wrapped_model(model: nn.Module) -> nn.Module:
    cur: nn.Module = model
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        while isinstance(cur, FSDP):
            cur = cur.module
    except Exception:
        pass
    if isinstance(cur, torch.nn.parallel.DistributedDataParallel):
        cur = cur.module
    return cur


def get_output_embedding_module(model: nn.Module) -> Optional[nn.Module]:
    m = _unwrap_wrapped_model(model)
    try:
        out = m.get_output_embeddings()
    except Exception:
        out = getattr(m, "lm_head", None)
    return out if isinstance(out, nn.Module) else None


def _is_layernorm(m: nn.Module) -> bool:
    return isinstance(
        m,
        (
            nn.LayerNorm,
            nn.GroupNorm,
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
        ),
    )


def _get_submodule(root: nn.Module, path: str) -> nn.Module:
    """
    Like nn.Module.get_submodule, but keeps a fallback for older stacks.
    Expects a dotted path, including ModuleList indices as integers (e.g. 'layers.0.fc1').
    """
    if hasattr(root, "get_submodule"):
        return root.get_submodule(path)  # type: ignore[attr-defined]

    cur: nn.Module = root
    for part in path.split("."):
        if part.isdigit():
            idx = int(part)
            if isinstance(cur, (nn.ModuleList, nn.Sequential, list, tuple)):
                cur = cur[idx]  # type: ignore[index]
            else:
                # ModuleList normally, but if it isn't indexable, fail loudly.
                raise AttributeError(f"Cannot index into {type(cur)} at '.{part}' for path='{path}'")
        else:
            nxt = getattr(cur, part, None)
            if not isinstance(nxt, nn.Module):
                raise AttributeError(f"Missing submodule '{part}' while resolving '{path}'")
            cur = nxt
    return cur


# -----------------------------------------------------------------------------
# Linear weight targets
# -----------------------------------------------------------------------------
def iter_linear_weight_modules(
    model: nn.Module,
    exclude_lm_head: bool = True,
) -> Iterator[Tuple[str, nn.Linear]]:
    m = _unwrap_wrapped_model(model)

    out_emb = get_output_embedding_module(m)
    exclude_ids: Set[int] = set()
    if exclude_lm_head and out_emb is not None:
        exclude_ids.add(id(out_emb))

    seen: Set[int] = set()

    for name, module in m.named_modules():
        if isinstance(module, nn.Embedding):
            continue
        if _is_layernorm(module):
            continue
        if exclude_lm_head and (name.endswith("lm_head") or id(module) in exclude_ids):
            continue

        if isinstance(module, nn.Linear):
            mid = id(module)
            if mid not in seen:
                seen.add(mid)
                yield name, module

        base = getattr(module, "base_layer", None)
        if isinstance(base, nn.Linear):
            if isinstance(base, nn.Embedding) or _is_layernorm(base):
                continue
            bid = id(base)
            if bid not in seen:
                seen.add(bid)
                yield f"{name}.base_layer", base


# -----------------------------------------------------------------------------
# Activation hook targets (same filter!)
# -----------------------------------------------------------------------------
def iter_activation_quant_modules(
    model: nn.Module,
    exclude_lm_head: bool = True,
) -> Iterator[Tuple[str, nn.Module]]:
    m = _unwrap_wrapped_model(model)

    out_emb = get_output_embedding_module(m)
    exclude_ids: Set[int] = set()
    if exclude_lm_head and out_emb is not None:
        exclude_ids.add(id(out_emb))

    seen: Set[int] = set()

    for name, module in m.named_modules():
        if isinstance(module, nn.Embedding):
            continue
        if _is_layernorm(module):
            continue
        if exclude_lm_head and (name.endswith("lm_head") or id(module) in exclude_ids):
            continue

        take = isinstance(module, nn.Linear)
        base = getattr(module, "base_layer", None)
        if isinstance(base, nn.Linear):
            take = True

        if take:
            mid = id(module)
            if mid not in seen:
                seen.add(mid)
                yield name, module


# -----------------------------------------------------------------------------
# Quantization kernels (LWC-aware for weights if enabled)
# -----------------------------------------------------------------------------
@torch.no_grad()
def quantize_weight_per_channel_absmax(
    w: torch.Tensor, n_bits: int, lwc_alpha: Optional[torch.Tensor] = None
) -> torch.Tensor:
    qmax = 2 ** (n_bits - 1) - 1
    scales = w.abs().max(dim=-1, keepdim=True)[0]  # [out, 1]

    if lwc_alpha is not None:
        a = lwc_alpha
        if a.dim() == 0:
            a = a.view(1, 1).expand_as(scales)
        elif a.shape != scales.shape:
            a = a.view(scales.shape)
        a = a.clamp(min=1e-8, max=1.0)
        scales = scales * a

    scales.clamp_(min=1e-5)
    scale = scales / qmax
    q = (w / scale).round().clamp(-qmax, qmax)
    return q * scale


@torch.no_grad()
def quantize_weight_per_tensor_absmax(
    w: torch.Tensor, n_bits: int, lwc_alpha: Optional[torch.Tensor] = None
) -> torch.Tensor:
    qmax = 2 ** (n_bits - 1) - 1
    s = w.abs().max()

    if lwc_alpha is not None:
        a = lwc_alpha.clamp(min=1e-8, max=1.0)
        s = s * a

    s = s.clamp(min=1e-5)
    scale = s / qmax
    q = (w / scale).round().clamp(-qmax, qmax)
    return q * scale


@torch.no_grad()
def quantize_activation_per_token_absmax(t: torch.Tensor, n_bits: int):
    if not isinstance(t, torch.Tensor):
        return t
    tq = t.clone()
    flat = tq.view(-1, tq.size(-1))
    scales = flat.abs().max(dim=-1, keepdim=True)[0]
    qmax = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(qmax)
    flat.div_(scales).round_().mul_(scales)
    return flat.view_as(tq)


@torch.no_grad()
def quantize_activation_per_tensor_absmax(t: torch.Tensor, n_bits: int):
    if not isinstance(t, torch.Tensor):
        return t
    tq = t.clone()
    scales = tq.abs().max()
    qmax = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(qmax)
    tq.div_(scales).round_().mul_(scales)
    return tq


# -----------------------------------------------------------------------------
# Weight + activation application
# -----------------------------------------------------------------------------
@torch.no_grad()
def quantize_linear_weight_in_place(
    linear: nn.Linear,
    weight_quant: str,
    n_bits: int,
    use_lwc: bool = False,
) -> None:
    w = linear.weight.data
    alpha = getattr(linear, "lwc_alpha", None) if use_lwc else None
    if weight_quant == "per_channel":
        linear.weight.data = quantize_weight_per_channel_absmax(w, n_bits, lwc_alpha=alpha)
    elif weight_quant == "per_tensor":
        linear.weight.data = quantize_weight_per_tensor_absmax(w, n_bits, lwc_alpha=alpha)
    else:
        raise ValueError(weight_quant)


@torch.no_grad()
def quantize_all_linear_weights(
    model: nn.Module,
    weight_quant: str,
    weight_bits: int,
    exclude_lm_head: bool,
    use_lwc: bool = False,
) -> None:
    for _, lin in iter_linear_weight_modules(model, exclude_lm_head):
        quantize_linear_weight_in_place(lin, weight_quant, weight_bits, use_lwc=use_lwc)


def make_activation_quant_fn(act_quant: str, act_bits: int):
    if act_quant == "per_token":
        return partial(quantize_activation_per_token_absmax, n_bits=act_bits)
    if act_quant == "per_tensor":
        return partial(quantize_activation_per_tensor_absmax, n_bits=act_bits)
    raise ValueError(act_quant)


def apply_act_fn_to_out(act_fn, out):
    if isinstance(out, torch.Tensor):
        return act_fn(out)
    if isinstance(out, (tuple, list)):
        return type(out)(apply_act_fn_to_out(act_fn, x) for x in out)
    return out


def add_activation_quant_hooks(
    model: nn.Module,
    act_quant: str,
    act_bits: int,
    act_location: str,
    exclude_lm_head: bool,
):
    act_fn = make_activation_quant_fn(act_quant, act_bits)
    hooks = []

    def pre_hook(_, inp):
        if inp and isinstance(inp[0], torch.Tensor):
            return (act_fn(inp[0]),) + inp[1:]
        return inp

    def fwd_hook(_, __, out):
        return apply_act_fn_to_out(act_fn, out)

    for _, module in iter_activation_quant_modules(model, exclude_lm_head):
        if act_location in {"input", "both"}:
            hooks.append(module.register_forward_pre_hook(pre_hook))
        if act_location in {"output", "both"}:
            hooks.append(module.register_forward_hook(fwd_hook))

    return hooks


def quantize_model(
    model: nn.Module,
    weight_quant: str,
    act_quant: str,
    quantize_bmm_input: bool,
    weight_bits: int,
    act_bits: int,
    act_location: str = "input",
    exclude_lm_head: bool = True,
    use_lwc: bool = False,
):
    """
    In-place fake quant:
      - weights are quantized once (and overwritten)
      - activations are quantized via forward hooks
    If use_lwc=True and a Linear has attribute/parameter `lwc_alpha`, it is used to shrink clipping.
    """
    _ = quantize_bmm_input
    quantize_all_linear_weights(model, weight_quant, weight_bits, exclude_lm_head, use_lwc=use_lwc)
    add_activation_quant_hooks(model, act_quant, act_bits, act_location, exclude_lm_head)
    return model


def register_lwc_alpha_params_from_state_dict(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    lwc_init: float = 1.0,
    exclude_lm_head: bool = True,
) -> List[str]:
    """
    Registers `lwc_alpha` Parameters *only* for modules that appear in the checkpoint state_dict.

    This is the critical piece for eval: without this, lwc_alpha keys are unexpected/ignored on load,
    and fake-quant never sees them.

    Returns the list of parameter keys registered (e.g. 'model.decoder.layers.0.fc1.lwc_alpha').
    """
    m = _unwrap_wrapped_model(model)
    out_emb = get_output_embedding_module(m) if exclude_lm_head else None
    out_emb_id = id(out_emb) if out_emb is not None else None

    alpha_keys = [k for k in state_dict.keys() if k.endswith("lwc_alpha")]
    registered: List[str] = []

    for k in alpha_keys:
        # module path is everything before '.lwc_alpha'
        if k.endswith(".lwc_alpha"):
            mod_path = k[: -len(".lwc_alpha")]
        else:
            # very unusual, but be safe
            continue

        # locate module
        try:
            mod = _get_submodule(m, mod_path)
        except Exception as e:
            raise RuntimeError(f"Checkpoint contains '{k}' but eval model has no submodule '{mod_path}'.") from e

        # skip output embedding / lm_head if requested
        if exclude_lm_head and out_emb_id is not None and id(mod) == out_emb_id:
            continue

        if not hasattr(mod, "weight"):
            raise RuntimeError(f"Checkpoint contains '{k}' but '{mod_path}' has no .weight attribute ({type(mod)}).")

        if hasattr(mod, "lwc_alpha"):
            # already registered; do not overwrite
            registered.append(k)
            continue

        # match ckpt shape if possible (most common is [out,1])
        ckpt_t = state_dict.get(k, None)
        shape = tuple(ckpt_t.shape) if isinstance(ckpt_t, torch.Tensor) else None

        # default shape for Linear is [out_features, 1]
        if shape is None:
            if isinstance(mod, nn.Linear):
                shape = (int(mod.out_features), 1)
            else:
                raise RuntimeError(f"Cannot infer lwc_alpha shape for '{mod_path}' ({type(mod)}).")

        init = torch.full(
            shape,
            float(lwc_init),
            dtype=getattr(mod.weight, "dtype", torch.float32),
            device=getattr(mod.weight, "device", torch.device("cpu")),
        )
        mod.register_parameter("lwc_alpha", nn.Parameter(init, requires_grad=True))
        registered.append(k)

    return registered
