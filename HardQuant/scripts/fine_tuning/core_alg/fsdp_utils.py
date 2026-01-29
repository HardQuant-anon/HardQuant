# fsdp_utils.py
#!/usr/bin/env python3
from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.distributed as dist

from torch.distributed.fsdp import BackwardPrefetch  # re-export usage consistency

from utils import DistState, dist_barrier, _dist_print, _get_decoder_layers, _unwrap_wrapped_model


def _fsdp_available() -> bool:
    try:
        import torch.distributed.fsdp  # noqa: F401
        return True
    except Exception:
        return False


def _get_fsdp_cls():
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    return FSDP


def is_fsdp_model(m: nn.Module) -> bool:
    if not _fsdp_available():
        return False
    FSDP = _get_fsdp_cls()
    return isinstance(m, FSDP)


def fsdp_get_full_state_dict_rank0(model: nn.Module, state: DistState) -> Optional[Dict[str, torch.Tensor]]:
    """
    Returns a CPU FULL state_dict on rank0 only, whether model is FSDP or not.

    IMPORTANT: If model is FSDP, ALL ranks must call this function whenever it is invoked,
    because producing a FULL_STATE_DICT performs collectives (all_gathers/unshards).
    """
    if not is_fsdp_model(model):
        if state.is_main:
            base = _unwrap_wrapped_model(model)
            return {k: v.detach().cpu() for k, v in base.state_dict().items()}
        return None

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig

    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    dist_barrier(state)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        sd = model.state_dict()
    dist_barrier(state)
    return sd if state.is_main else None


def fsdp_load_full_state_dict_rank0_only(
    model: nn.Module,
    state: DistState,
    sd_cpu_rank0: Optional[Dict[str, torch.Tensor]],
):
    """
    Load a FULL (unsharded) state_dict into an FSDP model, where only rank0 provides the dict.
    Non-rank0 ranks pass {}. This avoids materializing full weights on every rank.
    """
    if not is_fsdp_model(model):
        raise RuntimeError("fsdp_load_full_state_dict_rank0_only called on non-FSDP model.")

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig

    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    dist_barrier(state)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        if state.is_main:
            if sd_cpu_rank0 is None:
                raise RuntimeError("Rank0 must provide sd_cpu_rank0 to load full state dict.")
            model.load_state_dict(sd_cpu_rank0, strict=False)
        else:
            model.load_state_dict({}, strict=False)
    dist_barrier(state)


def broadcast_model_parameters_from_rank0(model: nn.Module, state: DistState) -> None:
    """
    For non-FSDP distributed runs: after rank0 loads weights, broadcast parameters to all ranks.
    """
    if (not state.enabled) or (not dist.is_initialized()):
        return
    with torch.no_grad():
        for p in model.parameters():
            if p is None:
                continue
            dist.broadcast(p.data, src=0)
    dist_barrier(state)


# -----------------------------------------------------------------------------
# Activation checkpointing (FSDP-friendly)
# -----------------------------------------------------------------------------
def _get_activation_ckpt_api():
    """Robustly import activation checkpointing APIs across PyTorch versions."""
    try:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (  # type: ignore
            apply_activation_checkpointing,
            checkpoint_wrapper,
            CheckpointImpl,
            CheckpointWrapper,
        )
        return apply_activation_checkpointing, checkpoint_wrapper, CheckpointImpl, CheckpointWrapper
    except Exception:
        pass

    try:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (  # type: ignore
            apply_activation_checkpointing_wrapper as apply_activation_checkpointing,
            checkpoint_wrapper,
            CheckpointImpl,
            CheckpointWrapper,
        )
        return apply_activation_checkpointing, checkpoint_wrapper, CheckpointImpl, CheckpointWrapper
    except Exception:
        pass

    raise RuntimeError(
        "Could not import PyTorch activation checkpointing wrapper APIs. "
        "Expected torch.distributed.algorithms._checkpoint.checkpoint_wrapper.{apply_activation_checkpointing, checkpoint_wrapper, CheckpointImpl}."
    )


def _maybe_apply_fsdp_activation_checkpointing(
    model: nn.Module,
    *,
    layer_cls: type,
    enabled: bool,
    offload_to_cpu: bool,
) -> None:
    if not enabled:
        return

    apply_activation_checkpointing, checkpoint_wrapper, CheckpointImpl, _CheckpointWrapper = _get_activation_ckpt_api()

    def wrapper_fn(m: nn.Module) -> nn.Module:
        try:
            return checkpoint_wrapper(
                m,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                offload_to_cpu=bool(offload_to_cpu),
            )
        except TypeError:
            return checkpoint_wrapper(m, checkpoint_impl=CheckpointImpl.NO_REENTRANT)

    def check_fn(m: nn.Module) -> bool:
        return isinstance(m, layer_cls)

    try:
        apply_activation_checkpointing(model, check_fn=check_fn, checkpoint_wrapper_fn=wrapper_fn)
        return
    except TypeError:
        pass

    apply_activation_checkpointing(model, wrapper_fn, check_fn)  # type: ignore[misc]


# -----------------------------------------------------------------------------
# FSDP wrapping
# -----------------------------------------------------------------------------
def wrap_with_fsdp_if_enabled(model: nn.Module, args, state: DistState) -> nn.Module:
    if (not state.enabled) or (not getattr(args, "fsdp", False)):
        return model
    if not _fsdp_available():
        raise RuntimeError("Requested --fsdp but torch.distributed.fsdp is not available.")

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy, CPUOffload
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    layers = _get_decoder_layers(model)
    layer_cls = type(layers[0])

    want_ckpt = bool(getattr(args, "gradient_checkpointing", False))
    ckpt_offload_to_cpu = bool(getattr(args, "fsdp_activation_ckpt_offload_to_cpu", False))
    if want_ckpt:
        _maybe_apply_fsdp_activation_checkpointing(
            model,
            layer_cls=layer_cls,
            enabled=True,
            offload_to_cpu=ckpt_offload_to_cpu,
        )

    mp = str(getattr(args, "fsdp_mixed_precision", "bf16")).lower()
    if mp == "bf16":
        param_dtype = reduce_dtype = buffer_dtype = torch.bfloat16
    elif mp == "fp16":
        param_dtype = reduce_dtype = buffer_dtype = torch.float16
    elif mp == "fp32":
        param_dtype = reduce_dtype = buffer_dtype = torch.float32
    else:
        raise ValueError(f"Invalid --fsdp-mixed-precision: {args.fsdp_mixed_precision}")
    mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

    sharding = str(getattr(args, "fsdp_sharding", "full_shard")).lower()
    if sharding == "full_shard":
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif sharding == "shard_grad_op":
        sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
    elif sharding == "no_shard":
        sharding_strategy = ShardingStrategy.NO_SHARD
    else:
        raise ValueError(f"Invalid --fsdp-sharding: {args.fsdp_sharding}")

    cpu_offload = CPUOffload(offload_params=bool(getattr(args, "fsdp_cpu_offload", False)))
    use_orig_params = bool(getattr(args, "fsdp_use_orig_params", False))

    try:
        _apply_activation_checkpointing, _checkpoint_wrapper, _CheckpointImpl, CheckpointWrapper = _get_activation_ckpt_api()
    except Exception:
        CheckpointWrapper = None  # type: ignore[assignment]

    def _is_checkpoint_wrapped_transformer_layer(m: nn.Module) -> bool:
        if CheckpointWrapper is None:
            return False
        if not isinstance(m, CheckpointWrapper):
            return False
        inner = getattr(m, "_checkpoint_wrapped_module", None)
        return isinstance(inner, layer_cls)

    def _policy(module, recurse, nonwrapped_numel):
        if _is_checkpoint_wrapped_transformer_layer(module):
            return (not recurse)
        try:
            return transformer_auto_wrap_policy(
                module=module,
                recurse=recurse,
                nonwrapped_numel=nonwrapped_numel,
                transformer_layer_cls={layer_cls},
                min_num_params=0,
            )
        except TypeError:
            return transformer_auto_wrap_policy(
                module=module,
                recurse=recurse,
                nonwrapped_numel=nonwrapped_numel,
                transformer_layer_cls={layer_cls},
            )

    model = model.to(state.device)
    kwargs = dict(
        auto_wrap_policy=_policy,
        mixed_precision=mixed_precision,
        sharding_strategy=sharding_strategy,
        cpu_offload=cpu_offload,
        device_id=state.local_rank if state.device.type == "cuda" else None,
        use_orig_params=use_orig_params,
        sync_module_states=True,
        forward_prefetch=False,
        backward_prefetch=BackwardPrefetch.BACKWARD_POST,
    )
    try:
        fsdp_model = FSDP(model, limit_all_gathers=True, **kwargs)
    except TypeError:
        fsdp_model = FSDP(model, **kwargs)

    if state.is_main:
        _dist_print(
            state,
            f"[FSDP] Enabled for student: sharding={sharding_strategy} mp={mp} use_orig_params={use_orig_params} "
            f"act_ckpt={want_ckpt} act_ckpt_offload_to_cpu={ckpt_offload_to_cpu}",
        )
    return fsdp_model
