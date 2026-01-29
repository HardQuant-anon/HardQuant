# utils.py
#!/usr/bin/env python3
from __future__ import annotations

import copy
import gc
import json
import math
import os
import random
import time
import types
import traceback as _traceback
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers.models.opt.modeling_opt import OPTPreTrainedModel  # type: ignore
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel  # type: ignore

try:
    from transformers.models.mistral.modeling_mistral import MistralPreTrainedModel  # type: ignore
except Exception:
    MistralPreTrainedModel = None  # type: ignore

try:
    from transformers.models.mixtral.modeling_mixtral import MixtralPreTrainedModel  # type: ignore
except Exception:
    MixtralPreTrainedModel = None  # type: ignore

try:
    from transformers.models.qwen3.modeling_qwen3 import Qwen3PreTrainedModel  # type: ignore
except Exception:
    Qwen3PreTrainedModel = None  # type: ignore

# Project imports (these are expected to be on sys.path by pipeline_distributed.py)
from eval import (  # type: ignore
    build_eval_loader,
    build_train_loader,
    perplexity,
    QuantParams,
    mean_activation_qerror_on_loader_inputs,
    mean_weight_qerror_across_layers,
)

from fake_quant import (  # type: ignore
    quantize_model as fake_quantize_model,
    get_output_embedding_module,
    iter_linear_weight_modules,
)

IN_DOMAIN_EVAL_FRACTION = 0.01
UNSEEN_EVAL_FRACTION = 0.01


# -----------------------------------------------------------------------------
# Distributed utilities
# -----------------------------------------------------------------------------
@dataclass
class DistState:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return (not self.enabled) or self.rank == 0


def _dist_print(state: DistState, *args, **kwargs):
    if state.is_main:
        print(*args, **kwargs)


def init_distributed_from_env(args) -> DistState:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    enabled = world_size > 1

    if enabled:
        if not dist.is_available():
            raise RuntimeError("torch.distributed is not available but WORLD_SIZE>1 was set.")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
    else:
        device = torch.device(getattr(args, "device", "cpu"))

    return DistState(enabled=enabled, rank=rank, world_size=world_size, local_rank=local_rank, device=device)


def dist_barrier(state: DistState):
    if state.enabled and dist.is_initialized():
        dist.barrier()


def dist_all_reduce_mean(state: DistState, t: torch.Tensor) -> torch.Tensor:
    if (not state.enabled) or (not dist.is_initialized()):
        return t
    tt = t.detach().clone()
    dist.all_reduce(tt, op=dist.ReduceOp.SUM)
    return tt / float(state.world_size)


def dist_all_reduce_sum(state: DistState, t: torch.Tensor) -> torch.Tensor:
    if (not state.enabled) or (not dist.is_initialized()):
        return t
    tt = t.detach().clone()
    dist.all_reduce(tt, op=dist.ReduceOp.SUM)
    return tt


def set_seed(seed: int, rank: int = 0):
    s = int(seed) + 1000 * int(rank)
    random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def maybe_wrap_train_loader_distributed(loader, state: DistState):
    if not state.enabled:
        return loader
    try:
        from torch.utils.data import DataLoader
        from torch.utils.data.distributed import DistributedSampler

        dataset = getattr(loader, "dataset", None)
        if dataset is None:
            return loader

        batch_size = getattr(loader, "batch_size", None) or 1
        num_workers = getattr(loader, "num_workers", 0)
        pin_memory = getattr(loader, "pin_memory", False)
        drop_last = getattr(loader, "drop_last", True)
        collate_fn = getattr(loader, "collate_fn", None)
        persistent_workers = getattr(loader, "persistent_workers", False)
        prefetch_factor = getattr(loader, "prefetch_factor", None)

        sampler = DistributedSampler(
            dataset,
            num_replicas=state.world_size,
            rank=state.rank,
            shuffle=True,
            drop_last=drop_last,
        )

        kwargs = dict(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=collate_fn,
            persistent_workers=persistent_workers,
        )
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor

        return DataLoader(**kwargs)
    except Exception:
        return loader


def maybe_wrap_eval_loader_distributed(loader, state: DistState):
    if not state.enabled:
        return loader
    try:
        from torch.utils.data import DataLoader
        from torch.utils.data.distributed import DistributedSampler

        dataset = getattr(loader, "dataset", None)
        if dataset is None:
            return loader

        batch_size = getattr(loader, "batch_size", None) or 1
        num_workers = getattr(loader, "num_workers", 0)
        pin_memory = getattr(loader, "pin_memory", False)
        drop_last = False
        collate_fn = getattr(loader, "collate_fn", None)
        persistent_workers = getattr(loader, "persistent_workers", False)
        prefetch_factor = getattr(loader, "prefetch_factor", None)

        sampler = DistributedSampler(
            dataset,
            num_replicas=state.world_size,
            rank=state.rank,
            shuffle=False,
            drop_last=drop_last,
        )

        kwargs = dict(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=collate_fn,
            persistent_workers=persistent_workers,
        )
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor

        return DataLoader(**kwargs)
    except Exception:
        return loader


def _jsonable(v):
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, torch.device):
        return str(v)
    if isinstance(v, (tuple, list)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(val) for k, val in v.items()}
    return v


def _jsonable_args(args) -> Dict[str, object]:
    return {k: _jsonable(v) for k, v in vars(args).items()}


def _write_results_json_rank0(state: DistState, output_dir: Path, results: Dict[str, object]) -> None:
    if not state.is_main:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    p = output_dir / "results.json"
    with p.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
        f.write("\n")


@torch.no_grad()
def distributed_perplexity(model: nn.Module, loader, state: DistState) -> Tuple[float, float, int]:
    model.eval()
    tot_nll_local = 0.0
    tot_tok_local = 0

    for (x,) in loader:
        x = x.to(state.device)
        out = model(input_ids=x, labels=x)
        tot_nll_local += float(out.loss) * x.numel()
        tot_tok_local += int(x.numel())

    nll_t = torch.tensor([tot_nll_local], device=state.device, dtype=torch.float64)
    tok_t = torch.tensor([tot_tok_local], device=state.device, dtype=torch.float64)

    nll_g = float(dist_all_reduce_sum(state, nll_t).item())
    tok_g = int(dist_all_reduce_sum(state, tok_t).item())

    avg_nll = nll_g / max(1, tok_g)
    ppl = math.exp(avg_nll)
    return ppl, avg_nll, tok_g


# -----------------------------------------------------------------------------
# Debug trace
# -----------------------------------------------------------------------------
def _now_ms() -> int:
    return int(time.time() * 1000)


def _safe_cuda_sync(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _safe_cuda_mem(device: torch.device) -> Dict[str, float]:
    if device.type != "cuda" or (not torch.cuda.is_available()):
        return {"alloc_mb": 0.0, "res_mb": 0.0, "peak_alloc_mb": 0.0, "peak_res_mb": 0.0}
    try:
        torch.cuda.synchronize(device)
    except Exception:
        pass
    alloc = float(torch.cuda.memory_allocated(device)) / (1024.0 * 1024.0)
    res = float(torch.cuda.memory_reserved(device)) / (1024.0 * 1024.0)
    peak_alloc = float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
    peak_res = float(torch.cuda.max_memory_reserved(device)) / (1024.0 * 1024.0)
    return {"alloc_mb": alloc, "res_mb": res, "peak_alloc_mb": peak_alloc, "peak_res_mb": peak_res}


class DebugTrace:
    def __init__(
        self,
        enabled: bool,
        state: DistState,
        output_dir: Optional[Path],
        sync: bool = True,
        sync_every: int = 1,
        mem: bool = True,
        flush: bool = True,
    ):
        self.enabled = bool(enabled)
        self.state = state
        self.sync = bool(sync)
        self.sync_every = int(max(1, sync_every))
        self.mem = bool(mem)
        self.flush = bool(flush)
        self._t0 = _now_ms()
        self._counter = 0
        self._fh = None

        if self.enabled and output_dir is not None:
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                p = output_dir / f"debug_rank{state.rank}_lr{state.local_rank}.log"
                self._fh = p.open("a", encoding="utf-8")
            except Exception:
                self._fh = None

    def close(self):
        if self._fh is not None:
            try:
                self._fh.flush()
                self._fh.close()
            except Exception:
                pass
        self._fh = None

    def _write(self, s: str):
        if not self.enabled:
            return
        print(s, flush=self.flush)
        if self._fh is not None:
            try:
                self._fh.write(s + "\n")
                if self.flush:
                    self._fh.flush()
            except Exception:
                pass

    def mark(self, msg: str, *, step: Optional[int] = None):
        if not self.enabled:
            return
        self._counter += 1
        do_sync = self.sync and ((self._counter % self.sync_every) == 0)
        if do_sync:
            _safe_cuda_sync(self.state.device)

        dt = _now_ms() - self._t0
        base = f"[DBG][+{dt}ms][r{self.state.rank}/lr{self.state.local_rank}]"
        if step is not None:
            base += f"[step={step}]"
        line = f"{base} {msg}"
        if self.mem:
            m = _safe_cuda_mem(self.state.device)
            line += (
                f" | mem alloc={m['alloc_mb']:.1f}MB res={m['res_mb']:.1f}MB "
                f"peak_alloc={m['peak_alloc_mb']:.1f}MB peak_res={m['peak_res_mb']:.1f}MB"
            )
        self._write(line)

        if do_sync:
            _safe_cuda_sync(self.state.device)

    def exception(self, where: str, exc: BaseException, *, step: Optional[int] = None):
        if not self.enabled:
            return
        dt = _now_ms() - self._t0
        hdr = f"[DBG][+{dt}ms][r{self.state.rank}/lr{self.state.local_rank}] EXCEPTION at {where}"
        if step is not None:
            hdr += f" (step={step})"
        self._write(hdr)
        self._write("".join(_traceback.format_exception(type(exc), exc, exc.__traceback__)))


@contextmanager
def dbg_region(trace: Optional[DebugTrace], name: str, *, step: Optional[int] = None):
    if trace is not None:
        trace.mark(f"ENTER {name}", step=step)
    try:
        yield
    except BaseException as e:
        if trace is not None:
            trace.exception(f"region {name}", e, step=step)
        raise
    finally:
        if trace is not None:
            trace.mark(f"EXIT {name}", step=step)


# -----------------------------------------------------------------------------
# Peak allocated memory tracer
# -----------------------------------------------------------------------------
def _mb(x_bytes: int) -> float:
    return float(x_bytes) / (1024.0 * 1024.0)


class PeakAllocTracker:
    def __init__(self, state: DistState, enabled: bool = True):
        self.state = state
        self.enabled = bool(enabled)
        self.best_alloc_bytes: int = 0
        self.best_where: str = ""
        self.best_step: Optional[int] = None

    def _cur_alloc_bytes(self) -> int:
        if (not self.enabled) or self.state.device.type != "cuda" or (not torch.cuda.is_available()):
            return 0
        try:
            torch.cuda.synchronize(self.state.device)
        except Exception:
            pass
        return int(torch.cuda.memory_allocated(self.state.device))

    def mark(self, where: str, *, step: Optional[int] = None) -> None:
        if (not self.enabled) or (not self.state.is_main):
            return
        cur = self._cur_alloc_bytes()
        if cur > self.best_alloc_bytes:
            self.best_alloc_bytes = cur
            self.best_where = str(where)
            self.best_step = step
            print(
                f"[PEAK_ALLOC] new_peak={_mb(cur):.1f}MB bytes={cur} at='{where}'"
                + (f" step={step}" if step is not None else ""),
                flush=True,
            )

    def summary(self) -> str:
        if not self.enabled:
            return "PeakAllocTracker disabled."
        return (
            f"peak_alloc={_mb(self.best_alloc_bytes):.1f}MB bytes={self.best_alloc_bytes} "
            f"at='{self.best_where}'"
            + (f" step={self.best_step}" if self.best_step is not None else "")
        )


# -----------------------------------------------------------------------------
# CUDA memory gathering
# -----------------------------------------------------------------------------
def _bytes_to_mb(x: float) -> float:
    return float(x) / (1024.0 * 1024.0)


def _cuda_mem_snapshot_bytes(device: torch.device) -> Dict[str, int]:
    if device.type != "cuda" or (not torch.cuda.is_available()):
        return {"cur_alloc": 0, "cur_reserved": 0, "peak_alloc": 0, "peak_reserved": 0}
    try:
        torch.cuda.synchronize(device)
    except Exception:
        pass
    return {
        "cur_alloc": int(torch.cuda.memory_allocated(device)),
        "cur_reserved": int(torch.cuda.memory_reserved(device)),
        "peak_alloc": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved": int(torch.cuda.max_memory_reserved(device)),
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


def dist_gather_cuda_mem_mb(state: DistState) -> Optional[List[Dict[str, float]]]:
    snap = _cuda_mem_snapshot_bytes(state.device)
    local = torch.tensor(
        [snap["cur_alloc"], snap["cur_reserved"], snap["peak_alloc"], snap["peak_reserved"], state.local_rank],
        device=state.device if state.device.type == "cuda" else "cpu",
        dtype=torch.int64,
    )

    if (not state.enabled) or (not dist.is_initialized()):
        return (
            [
                {
                    "rank": float(state.rank),
                    "local_rank": float(state.local_rank),
                    "cur_alloc_mb": _bytes_to_mb(snap["cur_alloc"]),
                    "cur_reserved_mb": _bytes_to_mb(snap["cur_reserved"]),
                    "peak_alloc_mb": _bytes_to_mb(snap["peak_alloc"]),
                    "peak_reserved_mb": _bytes_to_mb(snap["peak_reserved"]),
                }
            ]
            if state.is_main
            else None
        )

    gather_list = [torch.empty_like(local) for _ in range(state.world_size)]
    dist.all_gather(gather_list, local)

    if not state.is_main:
        return None

    out: List[Dict[str, float]] = []
    for r, t in enumerate(gather_list):
        cur_alloc, cur_reserved, peak_alloc, peak_reserved, lrank = [int(x) for x in t.tolist()]
        out.append(
            {
                "rank": float(r),
                "local_rank": float(lrank),
                "cur_alloc_mb": _bytes_to_mb(cur_alloc),
                "cur_reserved_mb": _bytes_to_mb(cur_reserved),
                "peak_alloc_mb": _bytes_to_mb(peak_alloc),
                "peak_reserved_mb": _bytes_to_mb(peak_reserved),
            }
        )
    return out


def format_mem_line(mem_list: List[Dict[str, float]]) -> str:
    parts = []
    for d in mem_list:
        r = int(d["rank"])
        lr = int(d["local_rank"])
        parts.append(
            f"r{r}(lr{lr}) peak_alloc={d['peak_alloc_mb']:.1f}MB peak_res={d['peak_reserved_mb']:.1f}MB "
            f"cur_alloc={d['cur_alloc_mb']:.1f}MB cur_res={d['cur_reserved_mb']:.1f}MB"
        )
    return " | ".join(parts)


# -----------------------------------------------------------------------------
# Model wrappers / access helpers
# -----------------------------------------------------------------------------
def _unwrap_wrapped_model(m: nn.Module) -> nn.Module:
    cur: nn.Module = m
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # type: ignore

        while isinstance(cur, FSDP):
            cur = cur.module  # type: ignore[assignment]
    except Exception:
        pass

    if isinstance(cur, torch.nn.parallel.DistributedDataParallel):
        cur = cur.module
    return cur


def _model_type(model: nn.Module) -> Optional[str]:
    m = _unwrap_wrapped_model(model)
    return getattr(getattr(m, "config", None), "model_type", None)


def _is_opt_family(model: nn.Module) -> bool:
    return _model_type(model) == "opt"


def _get_attr_path(root: nn.Module, path: List[str]) -> Optional[object]:
    cur: object = _unwrap_wrapped_model(root)
    for a in path:
        cur = getattr(cur, a, None)
        if cur is None:
            return None
    return cur


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
    paths = (opt_paths + llama_like_paths) if _is_opt_family(model) else (llama_like_paths + opt_paths)
    for p in paths:
        cur = _get_attr_path(model, p)
        if isinstance(cur, (list, nn.ModuleList)):
            return cur
    raise ValueError("Could not locate decoder layers.")


def _is_supported_causallm(model: nn.Module) -> bool:
    m = _unwrap_wrapped_model(model)

    if OPTPreTrainedModel is not None and isinstance(m, OPTPreTrainedModel):
        return True
    if LlamaPreTrainedModel is not None and isinstance(m, LlamaPreTrainedModel):
        return True
    if MistralPreTrainedModel is not None and isinstance(m, MistralPreTrainedModel):
        return True
    if MixtralPreTrainedModel is not None and isinstance(m, MixtralPreTrainedModel):
        return True
    if Qwen3PreTrainedModel is not None and isinstance(m, Qwen3PreTrainedModel):
        return True

    mt = _model_type(m)
    return mt in {"opt", "llama", "mistral", "mixtral", "qwen3"}


def _enable_gradient_checkpointing(model: nn.Module, fsdp_enabled: bool) -> None:
    base = _unwrap_wrapped_model(model)
    cfg = getattr(base, "config", None)
    if cfg is not None and hasattr(cfg, "use_cache"):
        try:
            cfg.use_cache = False
        except Exception:
            pass
    if hasattr(base, "enable_input_require_grads"):
        try:
            base.enable_input_require_grads()
        except Exception:
            pass

    # Only enable HF checkpointing when NOT using FSDP.
    if (not fsdp_enabled) and hasattr(base, "gradient_checkpointing_enable"):
        try:
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            base.gradient_checkpointing_enable()
        except Exception:
            pass


def maybe_freeze_lm_head(model: nn.Module, freeze: bool, state: Optional[DistState] = None) -> bool:
    if not freeze:
        return False
    out = get_output_embedding_module(model)
    if out is None:
        if state is None or state.is_main:
            print("[WARN] --freeze-lm-head was set but no output embedding module was found.", flush=True)
        return False
    any_frozen = False
    for p in out.parameters():
        if p.requires_grad:
            p.requires_grad_(False)
            any_frozen = True
    if (state is None) or state.is_main:
        w = getattr(out, "weight", None)
        print(f"[INFO] Freeze lm_head: {any_frozen} (weight.requires_grad={getattr(w, 'requires_grad', None)})", flush=True)
    return any_frozen


# -----------------------------------------------------------------------------
# Small utils
# -----------------------------------------------------------------------------
def cycle(loader: Iterable):
    while True:
        for batch in loader:
            yield batch


def _release_cuda_models(*models):
    for m in models:
        if m is not None:
            del m
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# -----------------------------------------------------------------------------
# Regularizer R_W
# -----------------------------------------------------------------------------
def compute_r_w(model: nn.Module, eps: float, exclude_lm_head: bool = True) -> torch.Tensor:
    out = get_output_embedding_module(model) if exclude_lm_head else None
    out_w = getattr(out, "weight", None) if out is not None else None
    out_w_id = id(out_w) if isinstance(out_w, nn.Parameter) else None

    device = next(model.parameters()).device
    total = None
    cnt = 0

    for _name, p in model.named_parameters():
        if out_w_id is not None and id(p) == out_w_id:
            continue
        if p.dim() < 2:
            continue
        W = p
        if W.size(-1) == 0:
            continue
        c = W.size(1)
        denom = W.pow(2).sum(dim=1).add(eps).sqrt()
        ratios = (W.abs().amax(dim=1) / (denom / (c**0.5))) ** 2
        s = ratios.sum()
        total = s if total is None else (total + s)
        cnt += ratios.numel()

    if total is None or cnt == 0:
        return torch.zeros((), device=device)
    return total / float(cnt)


# -----------------------------------------------------------------------------
# R_X tracker -- input activations to Linear layers
# -----------------------------------------------------------------------------
class RXTracker:
    def __init__(self, model: nn.Module, eps: float, use_smooth: bool = False, smooth_p: float = 8.0):
        self._device = next(model.parameters()).device
        self._eps = float(eps)
        self._use_smooth = bool(use_smooth)
        self._smooth_p = float(smooth_p)
        self._value: Optional[torch.Tensor] = None
        self._handles = []
        self._hooked_ids: set[int] = set()

        out_emb = get_output_embedding_module(model)
        out_emb_id = id(out_emb) if out_emb is not None else None

        def hook(_module, inp, _out):
            v = self._accumulate(inp)
            self._value = v if self._value is None else (self._value + v)

        def _maybe_register(m: nn.Module):
            mid = id(m)
            if mid in self._hooked_ids:
                return
            self._hooked_ids.add(mid)
            self._handles.append(m.register_forward_hook(hook))

        for name, module in model.named_modules():
            if (name.split(".")[-1] == "lm_head") or (out_emb_id is not None and id(module) == out_emb_id):
                continue

            base = getattr(module, "base_layer", None)
            if base is not None and (out_emb_id is not None and id(base) == out_emb_id):
                continue

            if isinstance(module, nn.Linear):
                _maybe_register(module)
            elif isinstance(base, nn.Linear):
                _maybe_register(base)

    def _accumulate(self, out) -> torch.Tensor:
        if isinstance(out, torch.Tensor):
            return self._compute_on_tensor(out)
        if isinstance(out, (list, tuple)):
            total = None
            for t in out:
                if isinstance(t, torch.Tensor):
                    v = self._compute_on_tensor(t)
                    total = v if total is None else (total + v)
            return torch.zeros((), device=self._device) if total is None else total
        return torch.zeros((), device=self._device)

    def _compute_on_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 1:
            return torch.zeros((), device=self._device)
        tensor = x.view(1, -1) if x.dim() == 1 else x.view(-1, x.size(-1))
        return self._compute_rows(tensor)

    def _compute_rows(self, x: torch.Tensor) -> torch.Tensor:
        c = x.size(-1)
        abs_x = x.abs()
        if self._use_smooth:
            maxv = abs_x.pow(self._smooth_p).sum(dim=-1, dtype=torch.float32).add(self._eps).pow(1.0 / self._smooth_p)
        else:
            maxv = abs_x.amax(dim=-1).to(torch.float32)

        l2 = (x * x).sum(dim=-1, dtype=torch.float32).add(self._eps).sqrt()
        denom = l2 / math.sqrt(float(c))

        ratios = (maxv / denom).pow(2)
        ratios = torch.nan_to_num(ratios, nan=0.0, posinf=0.0, neginf=0.0)
        ratios = ratios.clamp_(min=0.0, max=1e6)
        return ratios.mean(dtype=torch.float32)

    def reset(self):
        self._value = None

    def value(self) -> torch.Tensor:
        return torch.zeros((), device=self._device) if self._value is None else self._value

    def remove(self):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()
        self._hooked_ids.clear()


# -----------------------------------------------------------------------------
# SNOWS output capture
# -----------------------------------------------------------------------------
class LayerOutputCatcher:
    def __init__(self, model: nn.Module, layer_ids: List[int]):
        self._handles = []
        self.outputs: Dict[int, torch.Tensor] = {}
        layers = _get_decoder_layers(model)
        L = len(layers)
        wanted = sorted(set([i for i in layer_ids if 0 <= i < L]))

        def make_hook(i: int):
            def hook(_module, _inp, out):
                out0 = out[0] if isinstance(out, (tuple, list)) else out
                if isinstance(out0, torch.Tensor):
                    self.outputs[i] = out0

            return hook

        for i in wanted:
            self._handles.append(layers[i].register_forward_hook(make_hook(i)))

    def remove(self):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()


# -----------------------------------------------------------------------------
# QAT (+ optional LWC)
# -----------------------------------------------------------------------------
def round_ste(x: torch.Tensor) -> torch.Tensor:
    return (x.round() - x).detach() + x


def _quantize_weight_symmetric_ste(
    w: torch.Tensor,
    bits: int,
    weight_quant: str,
    group_size: int,
    lwc_alpha: Optional[torch.Tensor],
) -> torch.Tensor:
    if bits <= 0:
        return w
    qmax = (1 << (bits - 1)) - 1
    eps = 1e-8

    if weight_quant == "per_tensor":
        amax = w.abs().amax()
        alpha = lwc_alpha.clamp(min=eps, max=1.0) if lwc_alpha is not None else w.new_tensor(1.0)
        clip = (amax * alpha).clamp(min=eps)
        w_clip = w.clamp(-clip, clip)
        scale = clip / qmax
        q = round_ste(w_clip / scale).clamp(-qmax, qmax)
        return q * scale

    if weight_quant != "per_channel":
        raise ValueError(f"weight_quant must be per_channel|per_tensor, got {weight_quant}")

    out, in_feat = w.shape

    alpha_row = None
    if lwc_alpha is not None:
        alpha_row = lwc_alpha.clamp(min=eps, max=1.0)
        if alpha_row.dim() == 0:
            alpha_row = alpha_row.view(1, 1).expand(out, 1)
        elif alpha_row.shape != (out, 1):
            alpha_row = alpha_row.view(out, 1)

    if group_size is None or group_size <= 0:
        amax = w.abs().amax(dim=1, keepdim=True)
        clip = amax if alpha_row is None else (amax * alpha_row)
        clip = clip.clamp(min=eps)
        w_clip = w.clamp(-clip, clip)
        scale = clip / qmax
        q = round_ste(w_clip / scale).clamp(-qmax, qmax)
        return q * scale

    g = int(group_size)
    pad = (g - (in_feat % g)) % g
    w_pad = torch.cat([w, w.new_zeros(out, pad)], dim=1) if pad else w
    in2 = w_pad.shape[1]
    ng = in2 // g
    w_g = w_pad.view(out, ng, g)

    amax_g = w_g.abs().amax(dim=2, keepdim=True)
    clip = amax_g if alpha_row is None else (amax_g * alpha_row.view(out, 1, 1))
    clip = clip.clamp(min=eps)

    w_clip = w_g.clamp(-clip, clip)
    scale = clip / qmax
    q = round_ste(w_clip / scale).clamp(-qmax, qmax)
    wq = (q * scale).view(out, in2)
    return wq[:, :in_feat] if pad else wq


def enable_weight_qat_with_optional_lwc(
    model: nn.Module,
    bits_w: int,
    group_size: int,
    weight_quant: str,
    use_lwc: bool,
    lwc_init: float,
) -> None:
    for _name, m in iter_linear_weight_modules(model, exclude_lm_head=True):
        if getattr(m, "_qat_patched", False):
            continue

        if use_lwc and not hasattr(m, "lwc_alpha"):
            init = float(lwc_init)
            alpha = torch.full((m.out_features, 1), init, dtype=m.weight.dtype, device=m.weight.device)
            m.register_parameter("lwc_alpha", nn.Parameter(alpha, requires_grad=True))

        orig_forward = m.forward

        def new_forward(self: nn.Linear, x):
            lwc_alpha = getattr(self, "lwc_alpha", None) if use_lwc else None
            wq = _quantize_weight_symmetric_ste(
                self.weight,
                bits=int(bits_w),
                weight_quant=str(weight_quant),
                group_size=int(group_size),
                lwc_alpha=lwc_alpha,
            )
            return F.linear(x, wq, self.bias)

        m._qat_orig_forward = orig_forward  
        m.forward = types.MethodType(new_forward, m)
        m._qat_patched = True  

def _split_params_for_optimizer(model: nn.Module):
    lwc_params, other_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.endswith("lwc_alpha") or ".lwc_alpha" in n:
            lwc_params.append(p)
        else:
            other_params.append(p)
    return other_params, lwc_params


# -----------------------------------------------------------------------------
# Dataset helpers
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Rank0 eval helpers
# -----------------------------------------------------------------------------
def _nan() -> float:
    return float("nan")


def _eval_rank0_fp32_and_fakequant(
    *,
    model_id: str,
    device: torch.device,
    eval_loader_in_full,
    eval_loader_unseen_full,
    weight_quant: str,
    act_quant: str,
    quantize_bmm_input: bool,
    weight_bits: int,
    act_bits: int,
) -> Dict[str, float]:
    out = {
        "fp32_in_ppl": _nan(),
        "fp32_in_nll": _nan(),
        "fp32_unseen_ppl": _nan(),
        "fp32_unseen_nll": _nan(),
        "fq_in_ppl": _nan(),
        "fq_in_nll": _nan(),
        "fq_unseen_ppl": _nan(),
        "fq_unseen_nll": _nan(),
    }

    model = AutoModelForCausalLM.from_pretrained(model_id).to(device).eval()

    ppl_in, nll_in, _ = perplexity(model, eval_loader_in_full, str(device))  # type: ignore[arg-type]
    ppl_un, nll_un, _ = perplexity(model, eval_loader_unseen_full, str(device))  # type: ignore[arg-type]
    out["fp32_in_ppl"] = float(ppl_in)
    out["fp32_in_nll"] = float(nll_in)
    out["fp32_unseen_ppl"] = float(ppl_un)
    out["fp32_unseen_nll"] = float(nll_un)

    fake_quantize_model(
        model,
        weight_quant=weight_quant,
        act_quant=act_quant,
        quantize_bmm_input=quantize_bmm_input,
        weight_bits=weight_bits,
        act_bits=act_bits,
        exclude_lm_head=True,
    )
    model.eval()

    ppl_in, nll_in, _ = perplexity(model, eval_loader_in_full, str(device))  # type: ignore[arg-type]
    ppl_un, nll_un, _ = perplexity(model, eval_loader_unseen_full, str(device))  # type: ignore[arg-type]
    out["fq_in_ppl"] = float(ppl_in)
    out["fq_in_nll"] = float(nll_in)
    out["fq_unseen_ppl"] = float(ppl_un)
    out["fq_unseen_nll"] = float(nll_un)

    _release_cuda_models(model)
    return out


def _eval_rank0_fakequant_from_state_dict(
    *,
    model_id: str,
    device: torch.device,
    full_state_dict: Dict[str, torch.Tensor],
    eval_loader_in_full,
    eval_loader_unseen_full,
    weight_quant: str,
    act_quant: str,
    quantize_bmm_input: bool,
    weight_bits: int,
    act_bits: int,
) -> Dict[str, float]:
    out = {
        "fq_ft_in_ppl": _nan(),
        "fq_ft_in_nll": _nan(),
        "fq_ft_unseen_ppl": _nan(),
        "fq_ft_unseen_nll": _nan(),
    }

    model_fq = AutoModelForCausalLM.from_pretrained(model_id).to(device).eval()
    model_fq.load_state_dict(full_state_dict, strict=False)

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
        ppl_in, nll_in, _ = perplexity(model_fq, eval_loader_in_full, str(device))  # type: ignore[arg-type]
        ppl_un, nll_un, _ = perplexity(model_fq, eval_loader_unseen_full, str(device))  # type: ignore[arg-type]

    out["fq_ft_in_ppl"] = float(ppl_in)
    out["fq_ft_in_nll"] = float(nll_in)
    out["fq_ft_unseen_ppl"] = float(ppl_un)
    out["fq_ft_unseen_nll"] = float(nll_un)

    _release_cuda_models(model_fq)
    return out
