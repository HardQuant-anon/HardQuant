# /workspace/OmniQuant-main/models/int_qwen_layer.py
import math
import copy
from collections import OrderedDict
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from transformers.activations import ACT2FN

from quantize.int_linear import QuantLinear
from quantize.int_matmul import QuantMatMul
from quantize.omni_norm import OmniLlamaRMSNorm  # RMSNorm wrapper works fine for Qwen RMSNorm too
from models.transformation import *  # noqa: F403,F401


# -----------------------------------------------------------------------------
# Local RoPE helpers (avoid brittle transformers private imports)
# -----------------------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    # [..., d] where d is even
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.LongTensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compatible with the common HF RoPE shapes.

    q,k: [B, H, T, D]
    cos,sin often come as:
      - [T, D] or [1, 1, T, D] or [B, 1, T, D] or [1, T, D]
    position_ids: [B, T] or None

    We convert cos/sin into [B, 1, T, D] then apply:
      q' = q*cos + rotate_half(q)*sin
      k' = k*cos + rotate_half(k)*sin
    """
    # Normalize cos/sin to [B, 1, T, D]
    # Case 1: cos/sin already broadcastable
    if cos.dim() == 4:
        # could be [1,1,T,D] or [B,1,T,D]
        pass
    elif cos.dim() == 3:
        # [1,T,D] or [B,T,D] -> make [B,1,T,D]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    elif cos.dim() == 2:
        # [T,D] -> [1,1,T,D]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    else:
        raise RuntimeError(f"Unexpected cos shape: {tuple(cos.shape)}")

    # If position_ids provided and cos is only length-based, gather along T
    # Many HF rotary modules already return cos/sin aligned to positions, so
    # only gather if position_ids exists and cos has T dimension matching max positions.
    if position_ids is not None:
        # Ensure position_ids is [B,T]
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
        B, T = position_ids.shape
        # If cos is [1,1,seq,D] and seq >= T, gather per position
        if cos.shape[0] in (1, B) and cos.shape[2] >= T:
            # Expand batch if needed
            if cos.shape[0] == 1 and B > 1:
                cos = cos.expand(B, -1, -1, -1)
                sin = sin.expand(B, -1, -1, -1)
            # Gather along seq dimension
            idx = position_ids[:, None, :, None].expand(B, 1, T, cos.shape[-1])
            cos = torch.gather(cos, dim=2, index=idx)
            sin = torch.gather(sin, dim=2, index=idx)

    # Now cos/sin should broadcast to q,k
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat K/V heads for grouped-query attention.

    hidden_states: [B, H_kv, T, D]
    returns: [B, H_kv*n_rep, T, D]
    """
    if n_rep == 1:
        return hidden_states
    B, H_kv, T, D = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(B, H_kv, n_rep, T, D)
    return hidden_states.reshape(B, H_kv * n_rep, T, D)


# -----------------------------------------------------------------------------
# Minimal RotaryEmbedding fallback (only used if org_module.rotary_emb missing)
# -----------------------------------------------------------------------------
class _SimpleRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute up to max_pos for speed (can be extended if needed)
        t = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # [max_pos, dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)            # [max_pos, dim]
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)  # [1,1,T,D]
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)  # [1,1,T,D]

    def forward(self, x: torch.Tensor, position_ids: Optional[torch.LongTensor] = None, seq_len: Optional[int] = None):
        # Return (cos, sin) in a shape compatible with _apply_rotary_pos_emb.
        if seq_len is None:
            # infer from x which is [B, H, T, D] or [B, T, H, D] – we only need T
            seq_len = x.shape[-2]
        if seq_len > self.cos_cached.shape[2]:
            # Extend cache if needed (rare for your 2048 setup)
            t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            cos = emb.cos()[None, None, :, :]
            sin = emb.sin()[None, None, :, :]
            return cos.to(x.device, dtype=x.dtype), sin.to(x.device, dtype=x.dtype)
        return (
            self.cos_cached[:, :, :seq_len, :].to(x.device, dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, :].to(x.device, dtype=x.dtype),
        )


class QuantQwenMLP(nn.Module):
    def __init__(
        self,
        org_module: nn.Module,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        args=None,
    ):
        super().__init__()
        # Qwen2/Qwen3 HF naming matches Llama: gate_proj/up_proj/down_proj
        self.gate_proj = QuantLinear(org_module.gate_proj, args.weight_quant_params, args.act_quant_params)
        self.down_proj = QuantLinear(org_module.down_proj, args.weight_quant_params, args.act_quant_params)
        self.up_proj = QuantLinear(org_module.up_proj, args.weight_quant_params, args.act_quant_params)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class QuantQwenAttention(nn.Module):
    """Qwen2/Qwen3 attention is Llama-like: RoPE + q/k/v/o projections."""

    def __init__(self, org_module: nn.Module, config, args=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.num_key_value_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = getattr(config, "max_position_embeddings", 2048)

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got hidden_size={self.hidden_size}, num_heads={self.num_heads})."
            )

        self.rotary_emb = self._init_rotary_emb(org_module=org_module, config=config)

        self.k_proj = QuantLinear(org_module.k_proj, args.weight_quant_params, args.act_quant_params)
        self.v_proj = QuantLinear(org_module.v_proj, args.weight_quant_params, args.act_quant_params)
        self.q_proj = QuantLinear(org_module.q_proj, args.weight_quant_params, args.act_quant_params)
        self.o_proj = QuantLinear(org_module.o_proj, args.weight_quant_params, args.act_quant_params)

        self.qkt_matmul = QuantMatMul(args.q_quant_params, args.k_quant_params, matmul_func=torch.matmul)
        self.pv_matmul = QuantMatMul(args.p_quant_params, args.v_quant_params, matmul_func=torch.matmul)

        self.use_weight_quant = False
        self.use_act_quant = False

        import inspect
        print("=== DEBUG QWEN ATTN ===")
        print("type:", type(org_module))
        keys = sorted([k for k in dir(org_module) if not k.startswith("_")])
        print("attrs(sample):", keys[:80])
        for a in ("rotary_emb", "position_embeddings", "rope", "rotary", "inv_freq", "cos_cached", "sin_cached"):
            print(a, "->", getattr(org_module, a, None) is not None, "type", type(getattr(org_module, a, None)))
        print("named_modules tail:")
        nm = list(org_module.named_modules())
        for n, m in nm:
            print(" ", n, type(m))
        print("forward sig:", inspect.signature(org_module.forward))
        print("=======================")

    

    def _init_rotary_emb(self, org_module: nn.Module, config):
        """
        Qwen HF implementations vary across versions.
        Prefer reusing the exact RoPE module already attached to the HF attention module,
        but it may not be named rotary_emb.
    
        We try a small set of known attribute names used across Qwen/Llama variants.
        If none exist, fall back to a simple RoPE (last resort).
        """
        # 1) Common names across HF implementations
        for attr in ("rotary_emb", "rotary", "rope", "rope_emb", "position_embeddings"):
            if hasattr(org_module, attr):
                m = getattr(org_module, attr)
                if m is not None and isinstance(m, nn.Module):
                    return m
    
        # 2) Some implementations store it under a nested module
        # e.g., org_module.rotary_emb might be absent but org_module.q_proj has it (rare)
        for name, m in org_module.named_modules():
            if name.endswith(("rotary_emb", "position_embeddings", "rope")) and isinstance(m, nn.Module):
                return m
    
        base = getattr(config, "rope_theta", 10000.0)
        max_pos = getattr(config, "max_position_embeddings", 2048)
        return _SimpleRotaryEmbedding(dim=self.head_dim, max_position_embeddings=max_pos, base=base)

        
    def _default_position_ids(self, bsz: int, q_len: int, kv_seq_len: int, device: torch.device):
        start = max(int(kv_seq_len) - int(q_len), 0)
        pos = torch.arange(start, start + q_len, device=device, dtype=torch.long)
        return pos.unsqueeze(0).expand(bsz, q_len)

    def _rotary_cos_sin(self, value_states, position_ids, q_len, kv_seq_len):
    
        if position_ids is None:
            bsz = value_states.shape[0]
            position_ids = self._default_position_ids(bsz, q_len, kv_seq_len, value_states.device)
    
        # Try the two common call conventions
        try:
            return self.rotary_emb(value_states, seq_len=kv_seq_len)
        except TypeError:
            pass
        try:
            return self.rotary_emb(value_states, position_ids)
        except TypeError:
            pass
    
        # Fallback for our SimpleRotaryEmbedding
        return self.rotary_emb(value_states, position_ids=position_ids, seq_len=kv_seq_len)


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self._rotary_cos_sin(value_states=value_states, position_ids=position_ids, q_len=q_len, kv_seq_len=kv_seq_len)

        # Apply RoPE using local helper (no transformers private dependency)
        query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        key_states = _repeat_kv(key_states, self.num_key_value_groups)
        value_states = _repeat_kv(value_states, self.num_key_value_groups)

        query_states = self.qkt_matmul.quant_x1(query_states)
        key_states = self.qkt_matmul.quant_x2(key_states)
        attn_weights = self.qkt_matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights,
                torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device),
            )

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        attn_weights = self.pv_matmul.quant_x1(attn_weights)
        value_states = self.pv_matmul.quant_x2(value_states)
        attn_output = self.pv_matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, (QuantLinear, QuantMatMul)):
                m.set_quant_state(weight_quant, act_quant)


class QuantQwenDecoderLayer(nn.Module):
    def __init__(self, config, ori_layer, args):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = QuantQwenAttention(org_module=ori_layer.self_attn, config=config, args=args)
        self.mlp = QuantQwenMLP(
            org_module=ori_layer.mlp,
            hidden_size=self.hidden_size,
            intermediate_size=getattr(config, "intermediate_size", None) or config.hidden_size * 4,
            hidden_act=getattr(config, "hidden_act", "silu"),
            args=args,
        )

        self.input_layernorm = OmniLlamaRMSNorm(
            ori_layer.input_layernorm, eps=ori_layer.input_layernorm.variance_epsilon
        )
        self.post_attention_layernorm = OmniLlamaRMSNorm(
            ori_layer.post_attention_layernorm, eps=ori_layer.post_attention_layernorm.variance_epsilon
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=bool(output_attentions),
            use_cache=bool(use_cache),
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # Match HF decoder layer return convention
        outputs = (hidden_states,)
        if output_attentions:
            outputs = outputs + (self_attn_weights,)
        if use_cache:
            outputs = outputs + (present_key_value,)
        return outputs

    # The helper methods below are copied from OmniQuant llama layer logic.
    # They are used by quantize/omniquant.py for LET/LWC.
    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for _, m in self.named_modules():
            if isinstance(m, (QuantLinear, QuantMatMul)):
                m.set_quant_state(weight_quant, act_quant)

    def smooth_and_quant_temporary(self):
        if self.let:
            with torch.no_grad():
                for name, module in self.named_parameters():
                    if "smooth_scale" in name:
                        module.data = truncate_number(module)  # noqa: F403

            smooth_ln_fcs_temporary(  # noqa: F403
                self.input_layernorm,
                [self.self_attn.q_proj, self.self_attn.k_proj, self.self_attn.v_proj],
                self.qkv_smooth_scale,
                self.qkv_smooth_shift,
            )
            smooth_ln_fcs_temporary(  # noqa: F403
                self.post_attention_layernorm,
                [self.mlp.up_proj, self.mlp.gate_proj],
                self.fc1_smooth_scale,
                self.fc1_smooth_shift,
            )
            smooth_fc_fc_temporary(  # noqa: F403
                self.self_attn.v_proj,
                self.self_attn.o_proj,
                self.out_smooth_scale,
                self.out_smooth_shift,
            )
            smooth_q_k_temporary(  # noqa: F403
                self.self_attn.q_proj,
                self.self_attn.k_proj,
                self.qkt_smooth_scale,
            )
            self.mlp.down_proj.temp_weight = self.mlp.down_proj.weight
        else:
            for _, module in self.named_modules():
                if isinstance(module, QuantLinear):
                    module.temp_weight = module.weight

        for _, module in self.named_modules():
            if isinstance(module, QuantLinear):
                if hasattr(module, "temp_weight"):
                    module.temp_weight = module.weight_quantizer(module.temp_weight)
                else:
                    module.temp_weight = module.weight_quantizer(module.weight)
                if not hasattr(module, "temp_bias"):
                    module.temp_bias = module.bias
                module.use_temporary_parameter = True

    def clear_temp_variable(self):
        for _, module in self.named_modules():
            if isinstance(module, QuantLinear):
                del module.temp_weight
                del module.temp_bias

    @torch.no_grad()
    def smooth_and_quant_inplace(self):
        if self.let:
            for name, module in self.named_parameters():
                if "smooth_scale" in name:
                    module.data = truncate_number(module)  # noqa: F403

            smooth_ln_fcs_inplace(  # noqa: F403
                self.input_layernorm,
                [self.self_attn.q_proj, self.self_attn.k_proj, self.self_attn.v_proj],
                self.qkv_smooth_scale,
                self.qkv_smooth_shift,
            )
            smooth_ln_fcs_inplace(  # noqa: F403
                self.post_attention_layernorm,
                [self.mlp.up_proj, self.mlp.gate_proj],
                self.fc1_smooth_scale,
                self.fc1_smooth_shift,
            )
            smooth_fc_fc_inplace(  # noqa: F403
                self.self_attn.v_proj,
                self.self_attn.o_proj,
                self.out_smooth_scale,
                self.out_smooth_shift,
            )
            smooth_q_k_inplace(  # noqa: F403
                self.self_attn.q_proj,
                self.self_attn.k_proj,
                self.qkt_smooth_scale,
            )

        for _, module in self.named_modules():
            if isinstance(module, QuantLinear):
                module.weight = module.weight_quantizer(module.weight)
                module.use_temporary_parameter = False

    def let_parameters(self, use_shift=True):
        params = []
        template = "smooth" if use_shift else "smooth_scale"
        for n, m in self.named_parameters():
            if n.find(template) > -1:
                params.append(m)
        return iter(params)

    def lwc_parameters(self):
        params = []
        for n, m in self.named_parameters():
            if n.find("bound_factor") > -1:
                params.append(m)
        return iter(params)

    def omni_parameters(self, use_shift=True):
        params = []
        template = "smooth" if use_shift else "smooth_scale"
        for n, m in self.named_parameters():
            if n.find("bound_factor") > -1 or n.find(template) > -1:
                params.append(m)
        return iter(params)

    def omni_state_dict(self, destination=None, prefix="", keep_vars=False):
        if destination is None:
            destination = OrderedDict()
        for name, param in self.named_parameters():
            if name.find("smooth") > -1 or name.find("bound_factor") > -1:
                destination[prefix + name] = param if keep_vars else param.detach()
        return destination

    def register_scales_and_zeros(self):
        for _, module in self.named_modules():
            if isinstance(module, QuantLinear):
                module.weight_quantizer.register_scales_and_zeros()
