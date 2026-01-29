from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List

from transformers import AutoConfig
from transformers.optimization import Adafactor
from transformers import AutoModelForCausalLM

import torch
import torch.nn as nn
import torch.distributed as dist

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[2]
TOOLS_DIR = PROJECT_DIR / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from eval import (
    build_eval_loader,
    build_train_loader,
    perplexity,
    QuantParams,
    mean_activation_qerror_on_loader_inputs,
    mean_weight_qerror_across_layers,
)

from fake_quant import (
    quantize_model as fake_quantize_model,
)

from utils import (
    DistState,
    IN_DOMAIN_EVAL_FRACTION,
    UNSEEN_EVAL_FRACTION,
    _dist_print,
    init_distributed_from_env,
    dist_barrier,
    dist_all_reduce_mean,
    dist_all_reduce_sum,
    set_seed,
    maybe_wrap_train_loader_distributed,
    maybe_wrap_eval_loader_distributed,
    _jsonable_args,
    _write_results_json_rank0,
    distributed_perplexity,
    DebugTrace,
    dbg_region,
    PeakAllocTracker,
    dist_gather_cuda_mem_mb,
    format_mem_line,
    cycle,
    _release_cuda_models,
    _model_type,
    _get_decoder_layers,
    _unwrap_wrapped_model,
    _is_supported_causallm,
    _enable_gradient_checkpointing,
    maybe_freeze_lm_head,
    compute_r_w,
    RXTracker,
    LayerOutputCatcher,
    enable_weight_qat_with_optional_lwc,
    _split_params_for_optimizer,
    parse_dataset,
    hf_tuple,
    _make_tokenizer,
    _nan,
    _eval_rank0_fp32_and_fakequant,
    _eval_rank0_fakequant_from_state_dict,
)

from fsdp_utils import (
    is_fsdp_model,
    fsdp_get_full_state_dict_rank0,
    fsdp_load_full_state_dict_rank0_only,
    broadcast_model_parameters_from_rank0,
    wrap_with_fsdp_if_enabled,
)

DEFAULT_HPARAMS = {
    "model": "facebook/opt-1.3b",
    "dataset_in": "wikitext:wikitext-103-raw-v1",
    "dataset_unseen": "lambada:plain_text",
    "dataset_in_train_split": "train",
    "dataset_in_eval_split": "test",
    "dataset_unseen_eval_split": "validation",
    "block_size": 1024,
    "batch_size_eval": 1,
    "batch_size_ft": 1,
    "steps": 4000,
    "lr": 5e-5,
    "weight_decay": 0.0,
    "lambda_x": 5e-4,
    "lambda_w": 0.0,
    "lambda_factor": 2,
    "calib_tokens_train": 262144,
    "max_train_samples": None,
    "log_every": 10,
    "eval_every": 2000,
    "disable_diagnostics": False,
    "skip_initial_eval": False,
    "detect_anomaly": False,
    "rx_use_smooth": False,
    "rx_smooth_p": 8.0,
    "use_autoreg_eval": False,
    "gradient_checkpointing": False,
    "use_qat": False,
    "qat_bits_w": 4,
    "qat_group_size": 0,
    "use_lwc": False,
    "lwc_init": 1000,
    "lwc_lr": 1e-4,
    "weight_quant": "per_channel",
    "act_quant": "per_token",
    "quantize_bmm_input": False,
    "weight_bits": 4,
    "act_bits": 4,
    "eps": 1e-12,
    "loss_type": "layer_wise",
    "layer_wise_k": 80,
    "layer_wise_teacher_dtype": "fp16",
    "layer_wise_global_alpha": 0.000001,
    "layer_wise_teacher_rank": 1,
    "freeze_lm_head": False,
    "fsdp": False,
    "fsdp_sharding": "full_shard",
    "fsdp_mixed_precision": "bf16",
    "fsdp_cpu_offload": False,
    "fsdp_use_orig_params": False,
    "fsdp_auto_wrap": True,
    "fsdp_activation_ckpt_offload_to_cpu": False,
    "peak_mem_trace": False,
    "debug_trace": False,
    "debug_trace_sync": True,
    "debug_trace_sync_every": 1,
    "debug_trace_mem": True,
    "early_stop": False,
    "early_stop_every": 500,
    "early_stop_patience": 3,
    "early_stop_min_delta": 0.0,
    "early_stop_warmup": 0,
    "early_stop_fraction": 0.01,
}


def _reset_cuda_peak_stats(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass


def finetune(
    model: nn.Module,
    tok,
    eps: float,
    dataset_in: Tuple[str, Optional[str]],
    dataset_in_train_split: str,
    dataset_in_eval_split: str,
    dataset_unseen: Tuple[str, Optional[str]],
    dataset_unseen_eval_split: str,
    block_size: int,
    calib_tokens_train: int,
    batch_size_ft: int,
    batch_size_eval: int,
    steps: int,
    lr: float,
    weight_decay: float,
    lambda_x: float,
    lambda_w: float,
    lambda_factor: Optional[float],
    log_every: int,
    output_dir: Path,
    eval_every: int = 50,
    disable_diagnostics: bool = False,
    detect_anomaly: bool = False,
    rx_use_smooth: bool = False,
    rx_smooth_p: float = 8.0,
    weight_quant: str = "per_channel",
    act_quant: str = "per_token",
    quantize_bmm_input: bool = False,
    weight_bits: int = 8,
    act_bits: int = 8,
    loss_type: str = "ce",
    layer_wise_k: int = 2,
    layer_wise_teacher_dtype: str = "fp16",
    layer_wise_global_alpha: float = 0.0001,
    layer_wise_teacher_rank: int = 1,
    freeze_lm_head: bool = False,
    use_qat: bool = False,
    qat_bits_w: int = 4,
    qat_group_size: int = 0,
    use_lwc: bool = False,
    lwc_init: float = 1.0,
    lwc_lr: float = 0.0,
    max_train_samples: Optional[int] = None,
    dist_state: Optional[DistState] = None,
    fsdp_args: Optional[argparse.Namespace] = None,
    model_id_for_eval: Optional[str] = None,
    dbg: Optional[DebugTrace] = None,
    peak: Optional[PeakAllocTracker] = None,
    early_stop: bool = False,
    early_stop_every: int = 200,
    early_stop_patience: int = 3,
    early_stop_min_delta: float = 0.0,
    early_stop_warmup: int = 0,
    early_stop_fraction: float = 0.01,
) -> nn.Module:
    state = dist_state or DistState(False, 0, 1, 0, torch.device(next(model.parameters()).device))

    _layer_wise_dtype_str = str(layer_wise_teacher_dtype).lower()
    if _layer_wise_dtype_str == "fp16":
        layer_wise_wire_dtype = torch.float16
    elif _layer_wise_dtype_str == "bf16":
        layer_wise_wire_dtype = torch.bfloat16
    else:
        layer_wise_wire_dtype = torch.float32

    do_early_stop = bool(early_stop) and (int(early_stop_every) > 0) and (int(early_stop_patience) > 0)
    early_stop_every = int(max(1, early_stop_every))
    early_stop_patience = int(max(1, early_stop_patience))
    early_stop_min_delta = float(early_stop_min_delta)
    early_stop_warmup = int(max(0, early_stop_warmup))
    early_stop_fraction = float(early_stop_fraction)
    if not (0.0 < early_stop_fraction <= 1.0):
        raise ValueError(f"early_stop_fraction must be in (0,1], got {early_stop_fraction}")

    calib_tokens_train_eff = int(calib_tokens_train)
    max_train_samples_eff: Optional[int] = None
    if max_train_samples is not None:
        max_train_samples_eff = int(max_train_samples)
        if max_train_samples_eff <= 0:
            raise ValueError(f"max_train_samples must be > 0 when set, got {max_train_samples}")
        cap_tokens = int(max_train_samples_eff) * int(block_size)
        calib_tokens_train_eff = min(calib_tokens_train_eff, cap_tokens)

    loader = build_train_loader(
        tok,
        hf_tuple(dataset_in),
        block_size,
        calib_tokens_train_eff,
        batch_size_ft,
        split=dataset_in_train_split,
    )
    loader = maybe_wrap_train_loader_distributed(loader, state)
    iterator = cycle(loader)

    do_periodic_eval = eval_every is not None and eval_every > 0
    eval_loader_in_dist = eval_loader_unseen_dist = None
    eval_loader_in_full = eval_loader_unseen_full = None

    early_loader_train_full = None
    if do_early_stop and state.is_main:
        early_loader_train_full, _, _ = build_eval_loader(
            tok,
            hf_tuple(dataset_in),
            block_size,
            batch_size_eval,
            split=dataset_in_train_split,
            max_fraction=min(1.0, early_stop_fraction),
        )

    if do_periodic_eval:
        if state.enabled or state.is_main:
            eval_loader_in_dist, _, _ = build_eval_loader(
                tok,
                hf_tuple(dataset_in),
                block_size,
                batch_size_eval,
                split=dataset_in_eval_split,
                max_fraction=IN_DOMAIN_EVAL_FRACTION,
            )
            eval_loader_unseen_dist, _, _ = build_eval_loader(
                tok,
                hf_tuple(dataset_unseen),
                block_size,
                batch_size_eval,
                split=dataset_unseen_eval_split,
                max_fraction=UNSEEN_EVAL_FRACTION,
            )
            if state.enabled:
                eval_loader_in_dist = maybe_wrap_eval_loader_distributed(eval_loader_in_dist, state)
                eval_loader_unseen_dist = maybe_wrap_eval_loader_distributed(eval_loader_unseen_dist, state)
            if state.is_main:
                eval_loader_in_full, _, _ = build_eval_loader(
                    tok,
                    hf_tuple(dataset_in),
                    block_size,
                    batch_size_eval,
                    split=dataset_in_eval_split,
                    max_fraction=IN_DOMAIN_EVAL_FRACTION,
                )
                eval_loader_unseen_full, _, _ = build_eval_loader(
                    tok,
                    hf_tuple(dataset_unseen),
                    block_size,
                    batch_size_eval,
                    split=dataset_unseen_eval_split,
                    max_fraction=UNSEEN_EVAL_FRACTION,
                )
        dist_barrier(state)

    qp = QuantParams(bits_w=weight_bits, bits_x=act_bits, eps=eps)
    rx = RXTracker(model, eps, use_smooth=rx_use_smooth, smooth_p=rx_smooth_p)

    teacher: Optional[nn.Module] = None
    teacher_autocast = nullcontext()
    n_blocks = None
    phase_len = None
    n_phases = None

    if loss_type == "layer_wise":
        n_blocks = len(_get_decoder_layers(model))
        n_phases = max(1, n_blocks)
        phase_len = max(1, steps // n_phases)

        teacher_rank = int(layer_wise_teacher_rank)
        if state.enabled:
            teacher_rank = max(0, min(teacher_rank, state.world_size - 1))

        if (not state.enabled) or (state.rank == teacher_rank):
            teacher = copy.deepcopy(_unwrap_wrapped_model(model)).to(state.device).eval()
            for p in teacher.parameters():
                p.requires_grad_(False)

            if state.device.type == "cuda" and torch.cuda.is_available():
                if _layer_wise_dtype_str == "fp16":
                    teacher = teacher.to(dtype=torch.float16)
                    teacher_autocast = torch.amp.autocast("cuda", dtype=torch.float16)
                elif _layer_wise_dtype_str == "bf16":
                    teacher = teacher.to(dtype=torch.bfloat16)
                    teacher_autocast = torch.amp.autocast("cuda", dtype=torch.bfloat16)
                else:
                    teacher = teacher.to(dtype=torch.float32)
                    teacher_autocast = nullcontext()
            else:
                teacher = teacher.to(dtype=torch.float32)
                teacher_autocast = nullcontext()

    froze = maybe_freeze_lm_head(model, freeze=bool(freeze_lm_head), state=state)

    if use_qat:
        enable_weight_qat_with_optional_lwc(
            model,
            bits_w=int(qat_bits_w),
            group_size=int(qat_group_size),
            weight_quant=str(weight_quant),
            use_lwc=bool(use_lwc),
            lwc_init=float(lwc_init),
        )

    if fsdp_args is not None and bool(getattr(fsdp_args, "fsdp", False)):
        model = wrap_with_fsdp_if_enabled(model, fsdp_args, state)

    other_params, lwc_params = _split_params_for_optimizer(model)
    param_groups = []
    if other_params:
        param_groups.append({"params": other_params, "lr": float(lr), "weight_decay": float(weight_decay)})
    if lwc_params:
        lr_lwc = float(lr) if (lwc_lr is None or float(lwc_lr) == 0.0) else float(lwc_lr)
        param_groups.append({"params": lwc_params, "lr": lr_lwc, "weight_decay": 0.0})

    optimizer = Adafactor(
        param_groups,
        lr=float(lr),
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
    )

    metrics_path = output_dir / "train_metrics.jsonl"
    log_f = None
    if state.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        log_f = metrics_path.open("w", encoding="utf-8")

    _reset_cuda_peak_stats(state.device)
    dist_barrier(state)

    teacher_rank = int(layer_wise_teacher_rank)
    if state.enabled:
        teacher_rank = max(0, min(teacher_rank, state.world_size - 1))

    layer_wise_rel_eps = float(max(1e-6, float(eps)))

    def _rel_mse(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        sf = s.to(dtype=torch.float32)
        tf = t.to(dtype=torch.float32)
        num = (sf - tf).pow(2).mean()
        denom = tf.pow(2).mean().add(layer_wise_rel_eps)
        return num / denom

    lf: Optional[float] = None
    if lambda_factor is not None:
        lf = float(lambda_factor)

    lambda_x_eff_const = float(lambda_x)
    lambda_w_eff_const = float(lambda_w)
    x_init_batch: Optional[torch.Tensor] = None

    if lf is not None:
        model.train()
        (x0,) = next(iterator)
        x0 = x0.to(state.device)
        x_init_batch = x0

        with torch.no_grad():
            rx.reset()

            if loss_type == "ce":
                out0 = model(input_ids=x0, labels=x0, use_cache=False)
                main0 = out0.loss
                del out0
                rx0 = rx.value()
                rw0 = torch.zeros((), device=state.device)
                if float(lambda_w) != 0.0:
                    rw0 = compute_r_w(model, eps, exclude_lm_head=True)
            else:
                assert n_blocks is not None and n_phases is not None and phase_len is not None
                step0 = 0
                p_phase = min((step0 // phase_len) + 1, n_phases)
                start_block = min(max(int(p_phase) - 1, 0), n_blocks - 1)
                end_block = min(n_blocks - 1, start_block + int(layer_wise_k))
                target_blocks = list(range(start_block, end_block + 1))

                scatch = LayerOutputCatcher(model, target_blocks)
                s_out = model(input_ids=x0, use_cache=False)
                s_outs = {i: scatch.outputs[i] for i in target_blocks if i in scatch.outputs}
                s_logits = s_out.logits
                scatch.remove()
                del s_out

                if state.enabled and dist.is_initialized():
                    x_list = [torch.empty_like(x0) for _ in range(state.world_size)]
                    dist.all_gather(x_list, x0)
                else:
                    x_list = [x0]

                t_outs: Dict[int, torch.Tensor] = {}
                t_logits: Optional[torch.Tensor] = None

                if (not state.enabled) or (state.rank == teacher_rank):
                    if teacher is None:
                        raise RuntimeError(
                            "loss_type=layer_wise but teacher is None on teacher-hosting rank. "
                            f"(rank={state.rank}, teacher_rank={teacher_rank}, world_size={state.world_size})"
                        )

                    for src_rank, x_src in enumerate(x_list):
                        tcatch = LayerOutputCatcher(teacher, target_blocks)
                        with teacher_autocast:
                            t_out = teacher(input_ids=x_src, use_cache=False)

                        t_outs_rank_native = {i: tcatch.outputs[i].detach() for i in target_blocks if i in tcatch.outputs}
                        t_logits_rank_native = t_out.logits.detach()
                        tcatch.remove()
                        del t_out

                        if state.enabled and dist.is_initialized() and src_rank != teacher_rank:
                            for i in target_blocks:
                                if i not in t_outs_rank_native:
                                    raise RuntimeError(f"Teacher missing output for layer {i} (rank {src_rank}).")
                                dist.send(t_outs_rank_native[i].to(dtype=layer_wise_wire_dtype).contiguous(), dst=src_rank)
                            dist.send(t_logits_rank_native.to(dtype=layer_wise_wire_dtype).contiguous(), dst=src_rank)

                        if (not state.enabled) or (src_rank == teacher_rank):
                            t_outs = {
                                i: t_outs_rank_native[i].to(dtype=s_outs[i].dtype)
                                for i in target_blocks
                                if i in t_outs_rank_native and i in s_outs
                            }
                            t_logits = t_logits_rank_native.to(dtype=s_logits.dtype)
                else:
                    for i in target_blocks:
                        if i not in s_outs:
                            raise RuntimeError(f"Student missing output for layer {i}; cannot size recv buffer.")
                        buf_wire = torch.empty(s_outs[i].shape, device=state.device, dtype=layer_wise_wire_dtype)
                        dist.recv(buf_wire, src=teacher_rank)
                        t_outs[i] = buf_wire.to(dtype=s_outs[i].dtype)

                    t_logits_wire = torch.empty(s_logits.shape, device=state.device, dtype=layer_wise_wire_dtype)
                    dist.recv(t_logits_wire, src=teacher_rank)
                    t_logits = t_logits_wire.to(dtype=s_logits.dtype)

                layer_wise_local_rel0 = torch.zeros((), device=state.device, dtype=torch.float32)
                n_terms0 = 0
                for i in target_blocks:
                    if i not in s_outs or i not in t_outs:
                        continue
                    layer_wise_local_rel0 = layer_wise_local_rel0 + _rel_mse(s_outs[i], t_outs[i])
                    n_terms0 += 1
                if n_terms0 > 0:
                    layer_wise_local_rel0 = layer_wise_local_rel0 / float(n_terms0)

                assert t_logits is not None
                layer_wise_global_rel0 = _rel_mse(s_logits, t_logits)
                main0 = layer_wise_local_rel0 + float(layer_wise_global_alpha) * layer_wise_global_rel0

                rx0 = rx.value()
                rw0 = torch.zeros((), device=state.device)
                if float(lambda_w) != 0.0:
                    rw0 = compute_r_w(model, eps, exclude_lm_head=True)

            rx0 = torch.nan_to_num(rx0, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0, max=1e6)
            rw0 = torch.nan_to_num(rw0, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0, max=1e6)
            main0 = torch.nan_to_num(main0, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0, max=1e6)

            main0_m = dist_all_reduce_mean(state, main0.detach())
            rx0_m = dist_all_reduce_mean(state, rx0.detach())
            rw0_m = dist_all_reduce_mean(state, rw0.detach())

            enabled = []
            if float(lambda_x) != 0.0:
                enabled.append(("x", float(rx0_m.abs().clamp(min=1e-12).item())))
            if float(lambda_w) != 0.0:
                enabled.append(("w", float(rw0_m.abs().clamp(min=1e-12).item())))

            if enabled:
                target_total = float(main0_m.clamp(min=0.0).item()) * float(lf)
                share = target_total / float(len(enabled))
                for kind, denom in enabled:
                    lam = share / float(denom)
                    if kind == "x":
                        lambda_x_eff_const = lam
                    else:
                        lambda_w_eff_const = lam

        if state.is_main:
            _dist_print(
                state,
                f"[LAMBDA_FACTOR(init)] lambda_factor={float(lf)} "
                f"=> lambda_x_eff={lambda_x_eff_const:.6g} lambda_w_eff={lambda_w_eff_const:.6g}",
                flush=True,
            )
        dist_barrier(state)

    best_ckpt_path = output_dir / "best_state_dict.pt"
    best_fq_ppl_train: Optional[float] = None
    best_step: Optional[int] = None
    bad_checks = 0
    stopped_early = False

    def _broadcast_stop_flag(flag_rank0: bool) -> bool:
        if not state.enabled or not dist.is_initialized():
            return bool(flag_rank0)
        t = torch.tensor([1 if flag_rank0 else 0], device=state.device, dtype=torch.int32)
        dist.broadcast(t, src=0)
        return bool(int(t.item()) == 1)

    def _maybe_early_stop_check(step_idx1: int):
        nonlocal best_fq_ppl_train, best_step, bad_checks, stopped_early

        if not do_early_stop:
            return False, None, None
        if (step_idx1 % early_stop_every) != 0:
            return False, None, None

        dist_barrier(state)

        cur_ppl = None
        cur_nll = None

        full_sd = fsdp_get_full_state_dict_rank0(model, state)
        dist_barrier(state)

        if state.is_main:
            model_fq = AutoModelForCausalLM.from_pretrained(model_id_for_eval).to(state.device).eval()
            model_fq.load_state_dict(full_sd, strict=False)

            model_fq = fake_quantize_model(
                model_fq,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_bmm_input=quantize_bmm_input,
                weight_bits=weight_bits,
                act_bits=act_bits,
                exclude_lm_head=True,
            ).eval()

            with torch.no_grad():
                ppl_tr, nll_tr, _ = perplexity(model_fq, early_loader_train_full, str(state.device))

            cur_ppl = float(ppl_tr)
            cur_nll = float(nll_tr)
            _release_cuda_models(model_fq)

            warm_ok = step_idx1 >= early_stop_warmup
            if warm_ok:
                improved = False
                if best_fq_ppl_train is None:
                    improved = True
                else:
                    improved = cur_ppl < (best_fq_ppl_train - early_stop_min_delta)

                if improved:
                    best_fq_ppl_train = cur_ppl
                    best_step = step_idx1
                    bad_checks = 0
                    try:
                        torch.save(full_sd, best_ckpt_path)
                    except Exception as e:
                        _dist_print(state, f"[EARLY_STOP] WARNING: failed to save best checkpoint: {e}", flush=True)
                    _dist_print(state, f"[EARLY_STOP] new best train FQ PPL={cur_ppl:.6f} at step={step_idx1}", flush=True)
                else:
                    bad_checks += 1
                    _dist_print(
                        state,
                        f"[EARLY_STOP] no improve (train FQ PPL={cur_ppl:.6f}, best={best_fq_ppl_train:.6f} @step={best_step}), "
                        f"bad_checks={bad_checks}/{early_stop_patience}",
                        flush=True,
                    )

                if bad_checks >= early_stop_patience:
                    stopped_early = True
            else:
                _dist_print(
                    state,
                    f"[EARLY_STOP] warmup (step={step_idx1} < {early_stop_warmup}): train FQ PPL={cur_ppl:.6f} (not used for stopping)",
                    flush=True,
                )

        dist_barrier(state)
        should_stop = _broadcast_stop_flag(bool(stopped_early) if state.is_main else False)
        dist_barrier(state)
        return should_stop, cur_ppl, cur_nll

    for step in range(steps):
        with dbg_region(dbg, "train_step", step=step + 1):
            model.train()
            if peak is not None:
                peak.mark("train_step: start", step=step + 1)

            if step == 0 and x_init_batch is not None:
                x = x_init_batch
            else:
                (x,) = next(iterator)
                x = x.to(state.device)

            if peak is not None:
                peak.mark("after x.to(device)", step=step + 1)

            cm = torch.autograd.detect_anomaly() if detect_anomaly else nullcontext()
            with cm:
                rx.reset()

                if loss_type == "ce":
                    out = model(input_ids=x, labels=x, use_cache=False)
                    if peak is not None:
                        peak.mark("after student forward (ce)", step=step + 1)

                    main_loss = out.loss
                    ce_loss_for_log = main_loss.detach()
                    layer_wise_loss_for_log = torch.zeros((), device=state.device)

                    layer_wise_local_for_log = torch.zeros((), device=state.device)
                    layer_wise_global_for_log = torch.zeros((), device=state.device)

                else:
                    assert n_blocks is not None and n_phases is not None and phase_len is not None

                    p_phase = min((step // phase_len) + 1, n_phases)
                    start_block = min(max(int(p_phase) - 1, 0), n_blocks - 1)
                    end_block = min(n_blocks - 1, start_block + int(layer_wise_k))
                    target_blocks = list(range(start_block, end_block + 1))

                    scatch = LayerOutputCatcher(model, target_blocks)
                    s_out = model(input_ids=x, use_cache=False)
                    if peak is not None:
                        peak.mark("after student forward (layer_wise)", step=step + 1)

                    s_outs = {i: scatch.outputs[i] for i in target_blocks if i in scatch.outputs}
                    s_logits = s_out.logits
                    scatch.remove()
                    del s_out

                    if state.enabled and dist.is_initialized():
                        x_list = [torch.empty_like(x) for _ in range(state.world_size)]
                        dist.all_gather(x_list, x)
                    else:
                        x_list = [x]

                    t_outs: Dict[int, torch.Tensor] = {}
                    t_logits: Optional[torch.Tensor] = None

                    if (not state.enabled) or (state.rank == teacher_rank):
                        if teacher is None:
                            raise RuntimeError(
                                "loss_type=layer_wise but teacher is None on teacher-hosting rank during training. "
                                f"(rank={state.rank}, teacher_rank={teacher_rank}, world_size={state.world_size})"
                            )

                        for src_rank, x_src in enumerate(x_list):
                            tcatch = LayerOutputCatcher(teacher, target_blocks)
                            with torch.no_grad():
                                with teacher_autocast:
                                    t_out = teacher(input_ids=x_src, use_cache=False)

                            t_outs_rank_native = {i: tcatch.outputs[i].detach() for i in target_blocks if i in tcatch.outputs}
                            t_logits_rank_native = t_out.logits.detach()
                            tcatch.remove()
                            del t_out

                            if state.enabled and dist.is_initialized() and src_rank != teacher_rank:
                                for i in target_blocks:
                                    if i not in t_outs_rank_native:
                                        raise RuntimeError(f"Teacher missing output for layer {i} (rank {src_rank}).")
                                    dist.send(t_outs_rank_native[i].to(dtype=layer_wise_wire_dtype).contiguous(), dst=src_rank)
                                dist.send(t_logits_rank_native.to(dtype=layer_wise_wire_dtype).contiguous(), dst=src_rank)

                            if (not state.enabled) or (src_rank == teacher_rank):
                                t_outs = {
                                    i: t_outs_rank_native[i].to(dtype=s_outs[i].dtype)
                                    for i in target_blocks
                                    if i in t_outs_rank_native and i in s_outs
                                }
                                t_logits = t_logits_rank_native.to(dtype=s_logits.dtype)

                        if peak is not None:
                            peak.mark("after teacher forward (all ranks)", step=step + 1)

                    else:
                        for i in target_blocks:
                            if i not in s_outs:
                                raise RuntimeError(f"Student missing output for layer {i}; cannot size recv buffer.")
                            buf_wire = torch.empty(s_outs[i].shape, device=state.device, dtype=layer_wise_wire_dtype)
                            dist.recv(buf_wire, src=teacher_rank)
                            t_outs[i] = buf_wire.to(dtype=s_outs[i].dtype)

                        t_logits_wire = torch.empty(s_logits.shape, device=state.device, dtype=layer_wise_wire_dtype)
                        dist.recv(t_logits_wire, src=teacher_rank)
                        t_logits = t_logits_wire.to(dtype=s_logits.dtype)

                        if peak is not None:
                            peak.mark("after recv teacher tensors", step=step + 1)

                    layer_wise_local_rel = torch.zeros((), device=state.device, dtype=torch.float32)
                    n_terms = 0
                    for i in target_blocks:
                        if i not in s_outs or i not in t_outs:
                            continue
                        layer_wise_local_rel = layer_wise_local_rel + _rel_mse(s_outs[i], t_outs[i])
                        n_terms += 1
                    if n_terms > 0:
                        layer_wise_local_rel = layer_wise_local_rel / float(n_terms)

                    assert t_logits is not None
                    layer_wise_global_rel = _rel_mse(s_logits, t_logits)

                    main_loss = layer_wise_local_rel + float(layer_wise_global_alpha) * layer_wise_global_rel

                    ce_loss_for_log = torch.zeros((), device=state.device)
                    layer_wise_loss_for_log = main_loss.detach()
                    layer_wise_local_for_log = layer_wise_local_rel.detach()
                    layer_wise_global_for_log = layer_wise_global_rel.detach()

                rx_value = rx.value()
                rx_value = torch.nan_to_num(rx_value, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0, max=1e6)

                loss = main_loss + (lambda_x_eff_const * rx_value if lambda_x_eff_const != 0.0 else 0.0)

                rw_value = torch.zeros((), device=state.device)
                if lambda_w_eff_const != 0.0:
                    rw_value = compute_r_w(model, eps, exclude_lm_head=True)
                    loss = loss + lambda_w_eff_const * rw_value

                optimizer.zero_grad(set_to_none=True)
                if peak is not None:
                    peak.mark("after optimizer.zero_grad", step=step + 1)

                loss.backward()
                if peak is not None:
                    peak.mark("after backward", step=step + 1)

                optimizer.step()
                if peak is not None:
                    peak.mark("after optimizer.step", step=step + 1)

            loss_m = dist_all_reduce_mean(state, loss.detach())
            main_loss_m = dist_all_reduce_mean(state, main_loss.detach())
            ce_m = dist_all_reduce_mean(state, ce_loss_for_log.detach())
            layer_wise_m = dist_all_reduce_mean(state, layer_wise_loss_for_log.detach())
            rx_m = dist_all_reduce_mean(state, rx_value.detach())
            rw_m = dist_all_reduce_mean(state, rw_value.detach())

            layer_wise_local_m = dist_all_reduce_mean(state, layer_wise_local_for_log.detach())
            layer_wise_global_m = dist_all_reduce_mean(state, layer_wise_global_for_log.detach())

            record = {
                "step": step + 1,
                "train/loss": float(loss_m.cpu()),
                "train/loss_main": float(main_loss_m.cpu()),
                "train/CE": float(ce_m.cpu()),
                "train/LAYER_WISE": float(layer_wise_m.cpu()),
                "train/LAYER_WISE_local_rel": float(layer_wise_local_m.cpu()),
                "train/LAYER_WISE_global_rel": float(layer_wise_global_m.cpu()),
                "train/R_X": float(rx_m.cpu()),
                "train/R_W": float(rw_m.cpu()),
                "train/loss_type": str(loss_type),
                "layer_wise/global_alpha": float(layer_wise_global_alpha),
                "layer_wise/teacher_rank": int(teacher_rank),
                "layer_wise/rel_eps": float(layer_wise_rel_eps),
                "train/freeze_lm_head": bool(freeze_lm_head),
                "train/froze_lm_head": bool(froze),
                "train/use_qat": bool(use_qat),
                "train/qat_bits_w": int(qat_bits_w),
                "train/qat_group_size": int(qat_group_size),
                "train/use_lwc": bool(use_lwc),
                "train/lwc_init": float(lwc_init),
                "train/lwc_lr": float(lwc_lr),
                "train/lambda_factor": float(lf) if lf is not None else None,
                "train/lambda_x_eff": float(lambda_x_eff_const),
                "train/lambda_w_eff": float(lambda_w_eff_const),
                "train/max_train_samples": int(max_train_samples_eff) if max_train_samples_eff is not None else None,
                "train/calib_tokens_train_eff": int(calib_tokens_train_eff),
                "dist/enabled": bool(state.enabled),
                "dist/world_size": int(state.world_size),
                "dist/fsdp": bool(getattr(fsdp_args, "fsdp", False)) if fsdp_args is not None else False,
                "fsdp/act_ckpt": bool(getattr(fsdp_args, "gradient_checkpointing", False)) if fsdp_args is not None else False,
                "fsdp/act_ckpt_offload_to_cpu": bool(getattr(fsdp_args, "fsdp_activation_ckpt_offload_to_cpu", False))
                if fsdp_args is not None
                else False,
                "early_stop/enabled": bool(do_early_stop),
                "early_stop/every": int(early_stop_every) if do_early_stop else None,
                "early_stop/patience": int(early_stop_patience) if do_early_stop else None,
                "early_stop/min_delta": float(early_stop_min_delta) if do_early_stop else None,
                "early_stop/warmup": int(early_stop_warmup) if do_early_stop else None,
                "early_stop/fraction_train": float(early_stop_fraction) if do_early_stop else None,
            }

            if (step + 1) % log_every == 0 and (not disable_diagnostics) and state.is_main:
                if is_fsdp_model(model):
                    record["train/mean_w_qerr"] = float("nan")
                    record["train/mean_act_qerr"] = float("nan")
                else:
                    mean_w_qerr = float(mean_weight_qerror_across_layers(model, qp))
                    mean_act_qerr = (
                        float(mean_activation_qerror_on_loader_inputs(model, eval_loader_in_full, str(state.device), qp))
                        if eval_loader_in_full is not None
                        else float("nan")
                    )
                    record["train/mean_w_qerr"] = mean_w_qerr
                    record["train/mean_act_qerr"] = mean_act_qerr

            if do_periodic_eval and eval_every > 0 and (step + 1) % eval_every == 0:
                if eval_loader_in_dist is not None and eval_loader_unseen_dist is not None:
                    ppl_in, nll_in, _ = distributed_perplexity(model, eval_loader_in_dist, state)
                    ppl_un, nll_un, _ = distributed_perplexity(model, eval_loader_unseen_dist, state)
                    if state.is_main:
                        record.update(
                            {
                                "eval/in_nll": float(nll_in),
                                "eval/in_ppl": float(ppl_in),
                                "eval/unseen_nll": float(nll_un),
                                "eval/unseen_ppl": float(ppl_un),
                            }
                        )
                        _dist_print(
                            state,
                            f"[EVAL@{step+1}/{steps}] FP32 "
                            f"IN nll={float(nll_in):.4f} ppl={float(ppl_in):.4f} | "
                            f"UN nll={float(nll_un):.4f} ppl={float(ppl_un):.4f}",
                            flush=True,
                        )

                full_sd_eval = fsdp_get_full_state_dict_rank0(model, state)
                if state.is_main and eval_loader_in_full is not None and eval_loader_unseen_full is not None:
                    if model_id_for_eval is None:
                        raise RuntimeError("model_id_for_eval is required for fakequant periodic eval on rank0.")
                    if full_sd_eval is None:
                        raise RuntimeError("Could not obtain full state_dict on rank0 for fakequant periodic eval.")

                    model_fq = AutoModelForCausalLM.from_pretrained(model_id_for_eval).to(state.device).eval()
                    model_fq.load_state_dict(full_sd_eval, strict=False)

                    model_fq = fake_quantize_model(
                        model_fq,
                        weight_quant=weight_quant,
                        act_quant=act_quant,
                        quantize_bmm_input=quantize_bmm_input,
                        weight_bits=weight_bits,
                        act_bits=act_bits,
                        exclude_lm_head=True,
                    ).eval()

                    with torch.no_grad():
                        fq_ppl_in, fq_nll_in, _ = perplexity(model_fq, eval_loader_in_full, str(state.device))
                        fq_ppl_un, fq_nll_un, _ = perplexity(model_fq, eval_loader_unseen_full, str(state.device))
                    _release_cuda_models(model_fq)

                    record.update(
                        {
                            "eval_q/in_nll": float(fq_nll_in),
                            "eval_q/in_ppl": float(fq_ppl_in),
                            "eval_q/unseen_nll": float(fq_nll_un),
                            "eval_q/unseen_ppl": float(fq_ppl_un),
                        }
                    )
                    _dist_print(
                        state,
                        f"[EVAL_Q@{step+1}/{steps}] FQ   "
                        f"IN nll={float(fq_nll_in):.4f} ppl={float(fq_ppl_in):.4f} | "
                        f"UN nll={float(fq_nll_un):.4f} ppl={float(fq_ppl_un):.4f}",
                        flush=True,
                    )

            should_stop, cur_tr_ppl, cur_tr_nll = _maybe_early_stop_check(step + 1)
            if do_early_stop:
                record["early_stop/train_fq_ppl"] = float(cur_tr_ppl) if (cur_tr_ppl is not None) else float("nan")
                record["early_stop/train_fq_nll"] = float(cur_tr_nll) if (cur_tr_nll is not None) else float("nan")
                record["early_stop/best_train_fq_ppl"] = float(best_fq_ppl_train) if (best_fq_ppl_train is not None) else float("nan")
                record["early_stop/best_step"] = int(best_step) if (best_step is not None) else None
                record["early_stop/bad_checks"] = int(bad_checks)
                record["early_stop/stopped"] = bool(should_stop)

            if (step + 1) % log_every == 0:
                mem_list = dist_gather_cuda_mem_mb(state)
                if state.is_main and mem_list is not None:
                    record["mem/interval"] = {
                        f"rank{int(d['rank'])}": {
                            "local_rank": int(d["local_rank"]),
                            "peak_alloc_mb": float(d["peak_alloc_mb"]),
                            "peak_reserved_mb": float(d["peak_reserved_mb"]),
                            "cur_alloc_mb": float(d["cur_alloc_mb"]),
                            "cur_reserved_mb": float(d["cur_reserved_mb"]),
                        }
                        for d in mem_list
                    }
                    _dist_print(state, f"[MEM@{step+1}/{steps}] {format_mem_line(mem_list)}")
                _reset_cuda_peak_stats(state.device)

            if state.is_main and log_f is not None:
                log_f.write(json.dumps(record) + "\n")
                log_f.flush()

            if (step + 1) % log_every == 0:
                _dist_print(
                    state,
                    f"[FT@{step+1}/{steps}] loss={loss_m.item():.4f} main={main_loss_m.item():.4f} "
                    f"CE={float(ce_m):.4f} LAYER_WISE={float(layer_wise_m):.4f} "
                    f"LAYER_WISE_local_rel={float(layer_wise_local_m):.4f} LAYER_WISE_global_rel={float(layer_wise_global_m):.4f} "
                    f"R_X={rx_m.item():.4f} R_W={rw_m.item():.4f} "
                    f"lambda_x_eff={float(lambda_x_eff_const):.6g} lambda_w_eff={float(lambda_w_eff_const):.6g}",
                )

            if should_stop:
                if state.is_main:
                    _dist_print(
                        state,
                        f"[EARLY_STOP] STOPPING training loop at step={step+1}. "
                        f"best_train_fq_ppl={best_fq_ppl_train} best_step={best_step}",
                        flush=True,
                    )
                break

    if do_early_stop and best_ckpt_path.exists():
        if state.is_main:
            _dist_print(
                state,
                f"[EARLY_STOP] Restoring best checkpoint from {best_ckpt_path} "
                f"(best_step={best_step}, best_train_fq_ppl={best_fq_ppl_train})",
                flush=True,
            )
        dist_barrier(state)

        sd_cpu = None
        if state.is_main:
            try:
                sd_cpu = torch.load(best_ckpt_path, map_location="cpu")
            except Exception as e:
                _dist_print(state, f"[EARLY_STOP] WARNING: failed to load best checkpoint: {e}", flush=True)
                sd_cpu = None

        if is_fsdp_model(model):
            fsdp_load_full_state_dict_rank0_only(model, state, sd_cpu)
        else:
            if state.is_main and sd_cpu is not None:
                base = _unwrap_wrapped_model(model)
                base.load_state_dict(sd_cpu, strict=False)
            dist_barrier(state)
            broadcast_model_parameters_from_rank0(model, state)

    rx.remove()
    if state.is_main and log_f is not None:
        log_f.close()
    if teacher is not None:
        _release_cuda_models(teacher)

    return model.eval()


def build_parser(default_overrides: Optional[dict] = None) -> argparse.ArgumentParser:
    defaults = DEFAULT_HPARAMS.copy()
    if default_overrides:
        defaults.update(default_overrides)

    parser = argparse.ArgumentParser(description="HardQuant: Fine-tuning with hardness regularization.")
    parser.add_argument("--model", default=defaults["model"])
    parser.add_argument("--dataset-in", default=defaults["dataset_in"])
    parser.add_argument("--dataset-unseen", default=defaults["dataset_unseen"])
    parser.add_argument("--dataset-in-train-split", default=defaults["dataset_in_train_split"])
    parser.add_argument("--dataset-in-eval-split", default=defaults["dataset_in_eval_split"])
    parser.add_argument("--dataset-unseen-eval-split", default=defaults["dataset_unseen_eval_split"])
    parser.add_argument("--block-size", type=int, default=defaults["block_size"])
    parser.add_argument("--batch-size-eval", type=int, default=defaults["batch_size_eval"])
    parser.add_argument("--batch-size-ft", type=int, default=defaults["batch_size_ft"])
    parser.add_argument("--steps", type=int, default=defaults["steps"])
    parser.add_argument("--lr", type=float, default=defaults["lr"])
    parser.add_argument("--weight-decay", type=float, default=defaults["weight_decay"])
    parser.add_argument("--lambda-x", type=float, default=defaults["lambda_x"])
    parser.add_argument("--lambda-w", type=float, default=defaults["lambda_w"])
    parser.add_argument("--lambda-factor", type=float, default=defaults.get("lambda_factor", None))
    parser.add_argument("--calib-tokens-train", type=int, default=defaults["calib_tokens_train"])
    parser.add_argument("--max-train-samples", type=int, default=defaults.get("max_train_samples", None))
    parser.add_argument("--log-every", type=int, default=defaults["log_every"])
    parser.add_argument("--eval-every", type=int, default=defaults["eval_every"])
    parser.add_argument("--eps", type=float, default=defaults["eps"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default=None)

    parser.add_argument("--gradient-checkpointing", action="store_true", default=defaults.get("gradient_checkpointing", False))
    parser.add_argument("--disable-diagnostics", action="store_true", default=defaults["disable_diagnostics"])
    parser.add_argument("--skip-initial-eval", action="store_true", default=defaults.get("skip_initial_eval", False))
    parser.add_argument("--detect-anomaly", action="store_true", default=defaults.get("detect_anomaly", False))
    parser.add_argument("--use-autoreg-eval", action="store_true", default=defaults.get("use_autoreg_eval", False))
    parser.add_argument("--rx-use-smooth", action="store_true", default=defaults.get("rx_use_smooth", False))
    parser.add_argument("--rx-smooth-p", type=float, default=defaults.get("rx_smooth_p", 8.0))

    parser.add_argument("--weight-quant", type=str, default=defaults["weight_quant"], choices=["per_channel", "per_tensor"])
    parser.add_argument("--act-quant", type=str, default=defaults["act_quant"], choices=["per_token", "per_tensor"])
    parser.add_argument("--quantize-bmm-input", action="store_true", default=defaults["quantize_bmm_input"])
    parser.add_argument("--weight-bits", type=int, default=defaults["weight_bits"])
    parser.add_argument("--act-bits", type=int, default=defaults["act_bits"])

    # Renamed SNOWS -> layer_wise (naming only; loss stays the same)
    parser.add_argument("--loss-type", type=str, default=defaults.get("loss_type", "ce"), choices=["ce", "layer_wise"])
    parser.add_argument("--layer-wise-k", type=int, default=defaults.get("layer_wise_k", 2))
    parser.add_argument(
        "--layer-wise-teacher-dtype",
        type=str,
        default=defaults.get("layer_wise_teacher_dtype", "fp16"),
        choices=["fp32", "fp16", "bf16"],
    )
    parser.add_argument("--layer-wise-global-alpha", type=float, default=defaults.get("layer_wise_global_alpha", 1e-6))
    parser.add_argument("--layer-wise-teacher-rank", type=int, default=defaults.get("layer_wise_teacher_rank", 1))

    parser.add_argument("--freeze-lm-head", action="store_true", default=defaults.get("freeze_lm_head", False))
    parser.add_argument("--no-freeze-lm-head", dest="freeze_lm_head", action="store_false")
    parser.set_defaults(freeze_lm_head=defaults.get("freeze_lm_head", False))

    parser.add_argument("--use-qat", action="store_true", default=defaults.get("use_qat", False))
    parser.add_argument("--qat-bits-w", type=int, default=defaults.get("qat_bits_w", 4))
    parser.add_argument("--qat-group-size", type=int, default=defaults.get("qat_group_size", 0))
    parser.add_argument("--use-lwc", action="store_true", default=defaults.get("use_lwc", False))
    parser.add_argument("--lwc-init", type=float, default=defaults.get("lwc_init", 1.0))
    parser.add_argument("--lwc-lr", type=float, default=defaults.get("lwc_lr", 0.0))

    parser.add_argument("--early-stop", action="store_true", default=defaults.get("early_stop", False))
    parser.add_argument("--early-stop-every", type=int, default=defaults.get("early_stop_every", 200))
    parser.add_argument("--early-stop-patience", type=int, default=defaults.get("early_stop_patience", 3))
    parser.add_argument("--early-stop-min-delta", type=float, default=defaults.get("early_stop_min_delta", 0.0))
    parser.add_argument("--early-stop-warmup", type=int, default=defaults.get("early_stop_warmup", 0))
    parser.add_argument("--early-stop-fraction", type=float, default=defaults.get("early_stop_fraction", 0.01))

    output_default = defaults.get("output_dir") or (PROJECT_DIR / "outputs" / f"fakequant_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    parser.add_argument("--output-dir", type=Path, default=output_default)

    device_default = defaults.get("device") or ("cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--device", default=device_default)

    parser.add_argument("--fsdp", action="store_true", default=defaults.get("fsdp", False))
    parser.add_argument("--fsdp-sharding", type=str, default=defaults.get("fsdp_sharding", "full_shard"), choices=["full_shard", "shard_grad_op", "no_shard"])
    parser.add_argument("--fsdp-mixed-precision", type=str, default=defaults.get("fsdp_mixed_precision", "bf16"), choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--fsdp-cpu-offload", action="store_true", default=defaults.get("fsdp_cpu_offload", False))
    parser.add_argument("--fsdp-use-orig-params", action="store_true", default=defaults.get("fsdp_use_orig_params", False))
    parser.add_argument("--no-fsdp-use-orig-params", dest="fsdp_use_orig_params", action="store_false")
    parser.add_argument("--fsdp-auto-wrap", action="store_true", default=defaults.get("fsdp_auto_wrap", True))
    parser.add_argument("--no-fsdp-auto-wrap", dest="fsdp_auto_wrap", action="store_false")
    parser.add_argument("--fsdp-activation-ckpt-offload-to-cpu", action="store_true", default=defaults.get("fsdp_activation_ckpt_offload_to_cpu", False))

    parser.add_argument("--peak-mem-trace", action="store_true", default=defaults.get("peak_mem_trace", False))
    parser.add_argument("--debug-trace", action="store_true", default=defaults.get("debug_trace", False))
    parser.add_argument("--debug-trace-sync", action="store_true", default=defaults.get("debug_trace_sync", True))
    parser.add_argument("--no-debug-trace-sync", dest="debug_trace_sync", action="store_false")
    parser.add_argument("--debug-trace-sync-every", type=int, default=defaults.get("debug_trace_sync_every", 1))
    parser.add_argument("--debug-trace-mem", action="store_true", default=defaults.get("debug_trace_mem", True))
    parser.add_argument("--no-debug-trace-mem", dest="debug_trace_mem", action="store_false")

    return parser


def run_pipeline(args, state: DistState) -> dict:
    if state.is_main:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    dist_barrier(state)

    dbg = DebugTrace(
        enabled=bool(getattr(args, "debug_trace", False)),
        state=state,
        output_dir=args.output_dir if hasattr(args, "output_dir") else None,
        sync=bool(getattr(args, "debug_trace_sync", True)),
        sync_every=int(getattr(args, "debug_trace_sync_every", 1)),
        mem=bool(getattr(args, "debug_trace_mem", True)),
        flush=True,
    )
    peak = PeakAllocTracker(state, enabled=bool(getattr(args, "peak_mem_trace", False)))

    try:
        dbg.mark("run_pipeline: start")
        set_seed(args.seed, rank=state.rank)

        dataset_in = parse_dataset(args.dataset_in)
        dataset_unseen = parse_dataset(args.dataset_unseen)
        tok = _make_tokenizer(args.model)

        eval_loader_in_full = eval_loader_unseen_full = None

        if state.is_main:
            eval_loader_in_full, _, _ = build_eval_loader(
                tok,
                hf_tuple(dataset_in),
                args.block_size,
                args.batch_size_eval,
                split=args.dataset_in_eval_split,
                max_fraction=IN_DOMAIN_EVAL_FRACTION,
            )
        dist_barrier(state)

        if state.is_main:
            eval_loader_unseen_full, _, _ = build_eval_loader(
                tok,
                hf_tuple(dataset_unseen),
                args.block_size,
                args.batch_size_eval,
                split=args.dataset_unseen_eval_split,
                max_fraction=UNSEEN_EVAL_FRACTION,
            )
        dist_barrier(state)

        eval_loader_in_dist, _, _ = build_eval_loader(
            tok,
            hf_tuple(dataset_in),
            args.block_size,
            args.batch_size_eval,
            split=args.dataset_in_eval_split,
            max_fraction=IN_DOMAIN_EVAL_FRACTION,
        )
        if state.enabled:
            eval_loader_in_dist = maybe_wrap_eval_loader_distributed(eval_loader_in_dist, state)

        eval_loader_unseen_dist, _, _ = build_eval_loader(
            tok,
            hf_tuple(dataset_unseen),
            args.block_size,
            args.batch_size_eval,
            split=args.dataset_unseen_eval_split,
            max_fraction=UNSEEN_EVAL_FRACTION,
        )
        if state.enabled:
            eval_loader_unseen_dist = maybe_wrap_eval_loader_distributed(eval_loader_unseen_dist, state)

        dist_barrier(state)

        base_metrics = {
            "fp32_in_ppl": _nan(),
            "fp32_in_nll": _nan(),
            "fp32_unseen_ppl": _nan(),
            "fp32_unseen_nll": _nan(),
            "fq_in_ppl": _nan(),
            "fq_in_nll": _nan(),
            "fq_unseen_ppl": _nan(),
            "fq_unseen_nll": _nan(),
        }

        if args.use_autoreg_eval:
            _dist_print(state, "[INFO] Using autoregressive eval; skipping baseline PPL eval.")
        elif args.skip_initial_eval:
            _dist_print(state, "[INFO] Skipping baseline eval (--skip-initial-eval).")
        else:
            if state.is_main:
                if eval_loader_in_full is None or eval_loader_unseen_full is None:
                    raise RuntimeError("Expected rank0 full eval loaders.")
                base_metrics = _eval_rank0_fp32_and_fakequant(
                    model_id=args.model,
                    device=state.device,
                    eval_loader_in_full=eval_loader_in_full,
                    eval_loader_unseen_full=eval_loader_unseen_full,
                    weight_quant=args.weight_quant,
                    act_quant=args.act_quant,
                    quantize_bmm_input=args.quantize_bmm_input,
                    weight_bits=args.weight_bits,
                    act_bits=args.act_bits,
                )
                _dist_print(
                    state,
                    f"[Baseline FP32] IN nll={base_metrics['fp32_in_nll']:.4f} ppl={base_metrics['fp32_in_ppl']:.4f} | "
                    f"UN nll={base_metrics['fp32_unseen_nll']:.4f} ppl={base_metrics['fp32_unseen_ppl']:.4f}",
                )
                _dist_print(
                    state,
                    f"[Baseline FQ]   IN nll={base_metrics['fq_in_nll']:.4f} ppl={base_metrics['fq_in_ppl']:.4f} | "
                    f"UN nll={base_metrics['fq_unseen_nll']:.4f} ppl={base_metrics['fq_unseen_ppl']:.4f}",
                )
            dist_barrier(state)

        model_ft = AutoModelForCausalLM.from_pretrained(args.model).to(state.device)
        if not _is_supported_causallm(model_ft):
            raise ValueError(f"Expected OPT/LLaMA/Mistral-family. Got model_type={_model_type(model_ft)}")

        if bool(args.gradient_checkpointing):
            _enable_gradient_checkpointing(model_ft, fsdp_enabled=bool(args.fsdp))

        eval_every = 0 if args.use_autoreg_eval else args.eval_every
        if peak.enabled:
            peak.mark("before finetune() call")

        model_ft = finetune(
            model_ft,
            tok,
            args.eps,
            dataset_in,
            args.dataset_in_train_split,
            args.dataset_in_eval_split,
            dataset_unseen,
            args.dataset_unseen_eval_split,
            args.block_size,
            args.calib_tokens_train,
            args.batch_size_ft,
            args.batch_size_eval,
            args.steps,
            args.lr,
            args.weight_decay,
            args.lambda_x,
            args.lambda_w,
            args.lambda_factor,
            args.log_every,
            args.output_dir,
            eval_every=eval_every,
            disable_diagnostics=args.disable_diagnostics,
            detect_anomaly=args.detect_anomaly,
            rx_use_smooth=args.rx_use_smooth,
            rx_smooth_p=args.rx_smooth_p,
            weight_quant=args.weight_quant,
            act_quant=args.act_quant,
            quantize_bmm_input=args.quantize_bmm_input,
            weight_bits=args.weight_bits,
            act_bits=args.act_bits,
            loss_type=args.loss_type,
            layer_wise_k=args.layer_wise_k,
            layer_wise_teacher_dtype=args.layer_wise_teacher_dtype,
            layer_wise_global_alpha=args.layer_wise_global_alpha,
            layer_wise_teacher_rank=int(getattr(args, "layer_wise_teacher_rank", 1)),
            freeze_lm_head=bool(getattr(args, "freeze_lm_head", False)),
            use_qat=bool(getattr(args, "use_qat", False)),
            qat_bits_w=int(getattr(args, "qat_bits_w", 4)),
            qat_group_size=int(getattr(args, "qat_group_size", 0)),
            use_lwc=bool(getattr(args, "use_lwc", False)),
            lwc_init=float(getattr(args, "lwc_init", 1.0)),
            lwc_lr=float(getattr(args, "lwc_lr", 0.0)),
            max_train_samples=getattr(args, "max_train_samples", None),
            dist_state=state,
            fsdp_args=args,
            model_id_for_eval=args.model,
            dbg=dbg,
            peak=peak,
            early_stop=bool(getattr(args, "early_stop", False)),
            early_stop_every=int(getattr(args, "early_stop_every", 200)),
            early_stop_patience=int(getattr(args, "early_stop_patience", 3)),
            early_stop_min_delta=float(getattr(args, "early_stop_min_delta", 0.0)),
            early_stop_warmup=int(getattr(args, "early_stop_warmup", 0)),
            early_stop_fraction=float(getattr(args, "early_stop_fraction", 0.01)),
        )

        if peak.enabled:
            _dist_print(state, f"[PEAK_ALLOC_SUMMARY] {peak.summary()}")

        final_metrics: Dict[str, object] = {
            "ft_fp32_in_ppl": _nan(),
            "ft_fp32_in_nll": _nan(),
            "ft_fp32_unseen_ppl": _nan(),
            "ft_fp32_unseen_nll": _nan(),
            "ft_fq_in_ppl": _nan(),
            "ft_fq_in_nll": _nan(),
            "ft_fq_unseen_ppl": _nan(),
            "ft_fq_unseen_nll": _nan(),
        }

        if args.use_autoreg_eval:
            _dist_print(state, "[INFO] Using autoregressive eval; skipping final PPL eval.")
        else:
            dist_barrier(state)

            ppl_in_ft, nll_in_ft, _ = distributed_perplexity(model_ft, eval_loader_in_dist, state)
            ppl_un_ft, nll_un_ft, _ = distributed_perplexity(model_ft, eval_loader_unseen_dist, state)

            if state.is_main:
                final_metrics["ft_fp32_in_ppl"] = float(ppl_in_ft)
                final_metrics["ft_fp32_in_nll"] = float(nll_in_ft)
                final_metrics["ft_fp32_unseen_ppl"] = float(ppl_un_ft)
                final_metrics["ft_fp32_unseen_nll"] = float(nll_un_ft)
                _dist_print(
                    state,
                    f"[Final FT FP32] IN nll={float(nll_in_ft):.4f} ppl={float(ppl_in_ft):.4f} | "
                    f"UN nll={float(nll_un_ft):.4f} ppl={float(ppl_un_ft):.4f}",
                )

            dist_barrier(state)

            full_sd = fsdp_get_full_state_dict_rank0(model_ft, state)
            dist_barrier(state)

            if state.is_main:
                if eval_loader_in_full is None or eval_loader_unseen_full is None:
                    raise RuntimeError("Expected rank0 full eval loaders for final fakequant eval.")
                if full_sd is None:
                    raise RuntimeError("Could not obtain full state_dict on rank0 for final fakequant eval.")

                fqft = _eval_rank0_fakequant_from_state_dict(
                    model_id=args.model,
                    device=state.device,
                    full_state_dict=full_sd,
                    eval_loader_in_full=eval_loader_in_full,
                    eval_loader_unseen_full=eval_loader_unseen_full,
                    weight_quant=args.weight_quant,
                    act_quant=args.act_quant,
                    quantize_bmm_input=args.quantize_bmm_input,
                    weight_bits=args.weight_bits,
                    act_bits=args.act_bits,
                )

                final_metrics["ft_fq_in_ppl"] = float(fqft["fq_ft_in_ppl"])
                final_metrics["ft_fq_in_nll"] = float(fqft["fq_ft_in_nll"])
                final_metrics["ft_fq_unseen_ppl"] = float(fqft["fq_ft_unseen_ppl"])
                final_metrics["ft_fq_unseen_nll"] = float(fqft["fq_ft_unseen_nll"])

                _dist_print(
                    state,
                    f"[Final FT FQ]   IN nll={float(fqft['fq_ft_in_nll']):.4f} ppl={float(fqft['fq_ft_in_ppl']):.4f} | "
                    f"UN nll={float(fqft['fq_ft_unseen_nll']):.4f} ppl={float(fqft['fq_ft_unseen_ppl']):.4f}",
                )

            dist_barrier(state)

        repo_id = "RyanLucas3/Llama-3-ft-quant"
        ckpt_dir = args.output_dir / "hf_ckpt"

        full_sd = fsdp_get_full_state_dict_rank0(model_ft, state)
        dist_barrier(state)

        if state.is_main:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            if full_sd is None:
                raise RuntimeError("Could not obtain full state_dict on rank0 for saving.")

            cfg = AutoConfig.from_pretrained(args.model)
            model_to_save = AutoModelForCausalLM.from_config(cfg)
            model_to_save.load_state_dict(full_sd, strict=False)
            model_to_save.save_pretrained(ckpt_dir, safe_serialization=True)
            tok.save_pretrained(ckpt_dir)

            (args.output_dir / "hub_repo_id.txt").write_text(repo_id + "\n", encoding="utf-8")
            (args.output_dir / "hub_ckpt_dir.txt").write_text(str(ckpt_dir) + "\n", encoding="utf-8")

            del model_to_save
            gc.collect()

        dist_barrier(state)

        results: Dict[str, object] = {}
        if state.is_main:
            results.update({"seed": int(args.seed)})
            results.update(base_metrics)
            results.update(final_metrics)
            results["meta"] = {
                "timestamp": datetime.now().isoformat(),
                "dist_enabled": bool(state.enabled),
                "world_size": int(state.world_size),
                "rank": int(state.rank),
                "local_rank": int(state.local_rank),
            }
            results["args"] = _jsonable_args(args)
            _write_results_json_rank0(state, args.output_dir, results)

        dist_barrier(state)
        _release_cuda_models(model_ft)
        dist_barrier(state)

        dbg.mark("run_pipeline: end")
        return {} if not state.is_main else results

    finally:
        try:
            dbg.close()
        except Exception:
            pass


def main(default_overrides: Optional[dict] = None):
    parser = build_parser(default_overrides)
    args = parser.parse_args()

    state = init_distributed_from_env(args)

    if args.fsdp and not state.enabled:
        _dist_print(state, "[WARN] --fsdp set but WORLD_SIZE==1. FSDP will not be enabled.")

    if args.seeds is not None:
        seed_strs = [s.strip() for s in args.seeds.split(",") if s.strip()]
        seeds = [int(s) for s in seed_strs]

        if state.is_main:
            print(f"[MULTI-SEED][dist_enabled={state.enabled}] Running seeds: {seeds}")
        dist_barrier(state)

        for s in seeds:
            run_args = copy.deepcopy(args)
            run_args.seed = s
            run_args.output_dir = args.output_dir / f"seed_{s}"

            if state.is_main:
                print(f"\n===== Running seed {s} -> output_dir={run_args.output_dir} =====")
            dist_barrier(state)

            run_pipeline(run_args, state)
            dist_barrier(state)
    else:
        run_pipeline(args, state)

    if state.enabled and dist.is_initialized():
        dist_barrier(state)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
