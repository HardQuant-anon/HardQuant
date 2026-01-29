# models/transformation.py

import torch
import pdb


class TruncateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        truncated_tensor = input.clone()
        mask = truncated_tensor.abs() < threshold
        truncated_tensor[mask] = truncated_tensor[mask].sign() * threshold
        return truncated_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None


def truncate_number(number, threshold=1e-2):
    # avoid overflow with AMP training
    return TruncateFunction.apply(number, threshold)


# -----------------------------
# Helpers for Llama-3 GQA shapes
# -----------------------------
def _infer_head_dim_for_gqa(total_dim: int, kv_dim: int):
    """
    Infer head_dim such that:
      total_dim = num_heads * head_dim
      kv_dim    = num_kv_heads * head_dim
      num_heads % num_kv_heads == 0
    """
    for hd in (128, 64, 32, 16):
        if total_dim % hd != 0:
            continue
        if kv_dim % hd != 0:
            continue
        num_heads = total_dim // hd
        num_kv = kv_dim // hd
        if num_kv <= 0:
            continue
        if num_heads % num_kv != 0:
            continue
        return hd
    return None


def _project_gqa_vec_total_to_kv(vec_total: torch.Tensor, kv_dim: int, total_dim: int) -> torch.Tensor:
    """
    Project a vector defined on total_dim down to kv_dim by averaging across
    GQA repetition groups at head granularity.
    """
    v = vec_total.reshape(-1)
    if v.numel() != total_dim:
        raise RuntimeError(f"_project_gqa_vec_total_to_kv: expected {total_dim}, got {v.numel()}")

    hd = _infer_head_dim_for_gqa(total_dim=total_dim, kv_dim=kv_dim)
    if hd is None:
        g = total_dim // kv_dim
        if g > 0 and kv_dim * g == total_dim:
            return v.view(g, kv_dim).mean(dim=0).contiguous()
        return v[:kv_dim].contiguous()

    num_heads = total_dim // hd
    num_kv = kv_dim // hd
    group = num_heads // num_kv

    v = v.view(num_heads, hd)              # [num_heads, hd]
    v = v.view(num_kv, group, hd).mean(1)  # [num_kv, hd]
    return v.reshape(kv_dim).contiguous()


def _expand_gqa_vec_kv_to_total(vec_kv: torch.Tensor, kv_dim: int, total_dim: int) -> torch.Tensor:
    """
    Expand a vector defined on kv_dim up to total_dim by repeating across
    GQA groups at head granularity.
    """
    v = vec_kv.reshape(-1)
    if v.numel() != kv_dim:
        raise RuntimeError(f"_expand_gqa_vec_kv_to_total: expected {kv_dim}, got {v.numel()}")

    hd = _infer_head_dim_for_gqa(total_dim=total_dim, kv_dim=kv_dim)
    if hd is None:
        g = total_dim // kv_dim
        if g > 0 and kv_dim * g == total_dim:
            return v.repeat(g).contiguous()
        out = torch.zeros(total_dim, device=v.device, dtype=v.dtype)
        n = min(total_dim, kv_dim)
        out[:n] = v[:n]
        return out

    num_heads = total_dim // hd
    num_kv = kv_dim // hd
    group = num_heads // num_kv

    v = v.view(num_kv, hd)                       # [num_kv, hd]
    v = v[:, None, :].expand(num_kv, group, hd)  # [num_kv, group, hd]
    v = v.reshape(num_heads, hd)                 # [num_heads, hd]
    return v.reshape(total_dim).contiguous()


def _gqa_dual_scales_shifts(fc1, fc2, scales, shifts):
    """
    For smooth_fc_fc_*: produce
      (scales_kv, shifts_kv) for fc1 (v_proj, kv_dim)
      (scales_total, shifts_total) for fc2 (o_proj, total_dim)
    """
    if scales is None or shifts is None:
        raise RuntimeError("Expected non-None scales and shifts")

    scales = scales.reshape(-1)
    shifts = shifts.reshape(-1)

    kv_dim = int(getattr(fc1, "out_features", -1))
    total_dim = int(getattr(fc2, "in_features", -1))
    if kv_dim <= 0 or total_dim <= 0:
        return scales, shifts, scales, shifts

    if scales.numel() == total_dim and shifts.numel() == total_dim:
        scales_total, shifts_total = scales, shifts
        if total_dim % kv_dim == 0:
            scales_kv = _project_gqa_vec_total_to_kv(scales_total, kv_dim, total_dim)
            shifts_kv = _project_gqa_vec_total_to_kv(shifts_total, kv_dim, total_dim)
        else:
            scales_kv = scales_total[:kv_dim].contiguous()
            shifts_kv = shifts_total[:kv_dim].contiguous()
        return scales_kv, shifts_kv, scales_total, shifts_total

    if scales.numel() == kv_dim and shifts.numel() == kv_dim:
        scales_kv, shifts_kv = scales, shifts
        if total_dim % kv_dim == 0:
            scales_total = _expand_gqa_vec_kv_to_total(scales_kv, kv_dim, total_dim)
            shifts_total = _expand_gqa_vec_kv_to_total(shifts_kv, kv_dim, total_dim)
        else:
            scales_total = torch.zeros(total_dim, device=scales.device, dtype=scales.dtype)
            shifts_total = torch.zeros(total_dim, device=shifts.device, dtype=shifts.dtype)
            n = min(total_dim, kv_dim)
            scales_total[:n] = scales_kv[:n]
            shifts_total[:n] = shifts_kv[:n]
        return scales_kv, shifts_kv, scales_total, shifts_total

    # Mixed/unexpected: slice/pad defensively
    kv_dim = int(kv_dim)
    total_dim = int(total_dim)

    if scales.numel() >= kv_dim:
        scales_kv = scales[:kv_dim].contiguous()
    else:
        scales_kv = torch.nn.functional.pad(scales, (0, kv_dim - scales.numel()))

    if shifts.numel() >= kv_dim:
        shifts_kv = shifts[:kv_dim].contiguous()
    else:
        shifts_kv = torch.nn.functional.pad(shifts, (0, kv_dim - shifts.numel()))

    if scales.numel() >= total_dim:
        scales_total = scales[:total_dim].contiguous()
    else:
        scales_total = torch.nn.functional.pad(scales, (0, total_dim - scales.numel()))

    if shifts.numel() >= total_dim:
        shifts_total = shifts[:total_dim].contiguous()
    else:
        shifts_total = torch.nn.functional.pad(shifts, (0, total_dim - shifts.numel()))

    return scales_kv, shifts_kv, scales_total, shifts_total


def _qk_dual_scales(q_proj, k_proj, scales):
    """
    For smooth_q_k_*: produce
      scales_q: length q_proj.out_features (total_dim)
      scales_k: length k_proj.out_features (kv_dim)

    If provided scales is total_dim -> project to kv for k.
    If provided scales is kv_dim   -> expand to total for q.
    """
    s = scales.reshape(-1)

    q_dim = int(getattr(q_proj, "out_features", -1))
    k_dim = int(getattr(k_proj, "out_features", -1))
    if q_dim <= 0 or k_dim <= 0:
        return s, s

    if s.numel() == q_dim:
        scales_q = s
        if q_dim % k_dim == 0:
            scales_k = _project_gqa_vec_total_to_kv(scales_q, kv_dim=k_dim, total_dim=q_dim)
        else:
            scales_k = scales_q[:k_dim].contiguous()
        return scales_q, scales_k

    if s.numel() == k_dim:
        scales_k = s
        if q_dim % k_dim == 0:
            scales_q = _expand_gqa_vec_kv_to_total(scales_k, kv_dim=k_dim, total_dim=q_dim)
        else:
            scales_q = torch.zeros(q_dim, device=s.device, dtype=s.dtype)
            n = min(q_dim, k_dim)
            scales_q[:n] = scales_k[:n]
        return scales_q, scales_k

    # Unexpected: slice/pad to both
    if s.numel() >= q_dim:
        scales_q = s[:q_dim].contiguous()
    else:
        scales_q = torch.nn.functional.pad(s, (0, q_dim - s.numel()))

    if s.numel() >= k_dim:
        scales_k = s[:k_dim].contiguous()
    else:
        scales_k = torch.nn.functional.pad(s, (0, k_dim - s.numel()))

    return scales_q, scales_k


# -----------------------------
# Original transformation funcs
# -----------------------------
def smooth_ln_fcs_temporary(ln, fcs, scales, shifts):
    """
    TEMPORARY (non-inplace) smoothing that supports both LayerNorm(with bias) and RMSNorm(no bias).
    """
    ln.use_temporary_parameter = True
    if not isinstance(fcs, list):
        fcs = [fcs]

    scales = scales.reshape(-1)
    shifts = shifts.reshape(-1)

    # Biasless norms (e.g., Qwen3RMSNorm): ln.bias may not exist at all.
    if hasattr(ln, "bias") and (getattr(ln, "bias") is not None):
        ln.temp_bias = (ln.bias - shifts) / scales
    else:
        ln.temp_bias = (-1 * shifts) / scales

    ln.temp_weight = ln.weight / scales

    for fc in fcs:
        fc.use_temporary_parameter = True
        if hasattr(fc, "bias") and (getattr(fc, "bias") is not None):
            fc.temp_bias = fc.bias + fc.weight @ shifts
        else:
            fc.temp_bias = fc.weight @ shifts
        fc.temp_weight = fc.weight * scales.view(1, -1)


def smooth_fc_fc_temporary(fc1, fc2, scales, shifts=None):
    # only support for v_proj and out_proj now.
    fc1.use_temporary_parameter = True
    fc2.use_temporary_parameter = True

    if shifts is None:
        raise RuntimeError("smooth_fc_fc_temporary requires shifts (got None)")

    scales_kv, shifts_kv, scales_total, shifts_total = _gqa_dual_scales_shifts(fc1, fc2, scales, shifts)

    # fc1 uses KV vectors
    if hasattr(fc1, "temp_weight"):
        fc1.temp_bias = fc1.temp_bias - shifts_kv
        fc1.temp_bias = fc1.temp_bias / scales_kv.view(-1)
        fc1.temp_weight = fc1.temp_weight / scales_kv.view(-1, 1)
    else:
        fc1.temp_bias = fc1.bias / scales_kv.view(-1)
        fc1.temp_weight = fc1.weight / scales_kv.view(-1, 1)

    # fc2 uses total vectors
    if hasattr(fc2, "bias") and fc2.bias is not None:
        fc2.temp_bias = fc2.bias + fc2.weight @ shifts_total
    else:
        fc2.temp_bias = fc2.weight @ shifts_total
    fc2.temp_weight = fc2.weight * scales_total.view(1, -1)


def smooth_q_k_temporary(q_proj, k_proj, scales):
    q_proj.use_temporary_parameter = True
    k_proj.use_temporary_parameter = True

    scales_q, scales_k = _qk_dual_scales(q_proj, k_proj, scales)

    # q side uses scales_q
    q_proj.temp_weight = q_proj.temp_weight / scales_q.view(-1, 1)
    q_proj.temp_bias = q_proj.temp_bias / scales_q.view(-1)

    # k side uses scales_k
    k_proj.temp_weight = k_proj.temp_weight * scales_k.view(-1, 1)
    k_proj.temp_bias = k_proj.temp_bias * scales_k.view(-1)


def smooth_ln_fcs_inplace(ln, fcs, scales, shifts):
    """
    INPLACE smoothing that supports both LayerNorm(with bias) and RMSNorm(no bias).
    For biasless norms, we create a registered buffer called 'bias' (like original code intended),
    but we must not `del ln.bias` if it doesn't exist.
    """
    ln.use_temporary_parameter = False
    if not isinstance(fcs, list):
        fcs = [fcs]

    scales = scales.reshape(-1)
    shifts = shifts.reshape(-1)

    # ---- ln bias handling ----
    if hasattr(ln, "bias") and (getattr(ln, "bias") is not None):
        ln.bias.sub_(shifts)
        ln.bias.div_(scales)
    else:
        # If bias attribute exists, it might be None; if it doesn't exist (Qwen3RMSNorm), avoid del.
        if hasattr(ln, "bias"):
            try:
                del ln.bias
            except Exception:
                pass
        # register a bias buffer (so later code that expects ln.bias can still work)
        ln.register_buffer("bias", (-1 * shifts) / scales)

    ln.weight.div_(scales)

    # ---- fc bias handling ----
    for fc in fcs:
        fc.use_temporary_parameter = False
        if hasattr(fc, "bias") and (getattr(fc, "bias") is not None):
            fc.bias.add_(fc.weight @ shifts)
        else:
            if hasattr(fc, "bias"):
                try:
                    del fc.bias
                except Exception:
                    pass
            fc.register_buffer("bias", fc.weight @ shifts)
        fc.weight.mul_(scales.view(1, -1))


def smooth_fc_fc_inplace(fc1, fc2, scales, shifts=None):
    # only support for v_proj and out_proj now.
    fc1.use_temporary_parameter = False
    fc2.use_temporary_parameter = False

    if shifts is None:
        raise RuntimeError("smooth_fc_fc_inplace requires shifts (got None)")

    scales_kv, shifts_kv, scales_total, shifts_total = _gqa_dual_scales_shifts(fc1, fc2, scales, shifts)

    # fc1 uses KV vectors
    fc1.bias.sub_(shifts_kv)
    fc1.bias.div_(scales_kv.view(-1))
    fc1.weight.div_(scales_kv.view(-1, 1))

    # fc2 uses total vectors
    if hasattr(fc2, "bias") and fc2.bias is not None:
        fc2.bias.add_(fc2.weight @ shifts_total)
    else:
        if hasattr(fc2, "bias"):
            try:
                del fc2.bias
            except Exception:
                pass
        fc2.register_buffer("bias", fc2.weight @ shifts_total)
    fc2.weight.mul_(scales_total.view(1, -1))


def smooth_q_k_inplace(q_proj, k_proj, scales):
    q_proj.use_temporary_parameter = False
    k_proj.use_temporary_parameter = False

    scales_q, scales_k = _qk_dual_scales(q_proj, k_proj, scales)

    q_proj.weight.div_(scales_q.view(-1, 1))
    q_proj.bias.div_(scales_q.view(-1))

    k_proj.weight.mul_(scales_k.view(-1, 1))
    k_proj.bias.mul_(scales_k.view(-1))
