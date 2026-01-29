import torch
import torch.nn as nn

"""
Modify normalization layer to adapt the training of learnable equivalent transformation.

Key fix:
- PyTorch F.layer_norm requires input/weight/bias to have consistent dtype.
- In OmniQuant runs you can end up with x in fp16/bf16 while LN weight/bias are fp32.
- We compute LN in fp32 for numerical stability and to avoid dtype mismatch, then cast back.
"""


class OmniLayerNorm(nn.Module):
    def __init__(self, ori_layer_norm) -> None:
        super().__init__()
        self.use_act_quant = True

        # These are buffers in this repo (not Parameters). Keep that behavior.
        # Note: register_buffer keeps dtype/device tracking but they are not trainable.
        self.register_buffer("weight", ori_layer_norm.weight)
        if ori_layer_norm.bias is not None:
            self.register_buffer("bias", ori_layer_norm.bias)
        else:
            self.bias = None

        self.eps = ori_layer_norm.eps
        self.norm_func = nn.functional.layer_norm
        self.normalized_shape = ori_layer_norm.normalized_shape
        self.use_temporary_parameter = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Choose current affine params
        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias
        else:
            weight = self.weight
            bias = self.bias

        # --- DTYPE FIX ---
        # Compute LN in fp32 to avoid "expected Float but found Half" and to improve stability.
        orig_dtype = x.dtype
        x_fp32 = x.float()

        weight_fp32 = weight.float() if weight is not None else None
        bias_fp32 = bias.float() if bias is not None else None

        out = self.norm_func(x_fp32, self.normalized_shape, weight_fp32, bias_fp32, eps=self.eps)
        return out.to(orig_dtype)

    def set_quant_state(self, use_weight_quant, use_act_quant):
        self.use_act_quant = use_act_quant


class OmniLlamaRMSNorm(nn.Module):
    def __init__(self, ori_norm, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm.

        This implementation already computes the variance in fp32 and casts back.
        We additionally make sure the multiplicative weight (and optional bias) are applied
        in fp32 to avoid mixed-dtype issues when temporary parameters are used.
        """
        super().__init__()
        self.register_buffer("weight", ori_norm.weight)
        self.bias = None
        self.variance_epsilon = eps
        self.use_temporary_parameter = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype

        # RMSNorm core in fp32
        hs_fp32 = hidden_states.float()
        variance = hs_fp32.pow(2).mean(-1, keepdim=True)
        hs_fp32 = hs_fp32 * torch.rsqrt(variance + self.variance_epsilon)

        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = getattr(self, "temp_bias", None)
        else:
            weight = self.weight
            bias = self.bias if hasattr(self, "bias") else None

        w_fp32 = weight.float() if weight is not None else None
        if bias is not None:
            b_fp32 = bias.float()
            out_fp32 = w_fp32 * hs_fp32 + b_fp32
        else:
            out_fp32 = w_fp32 * hs_fp32

        return out_fp32.to(orig_dtype)
