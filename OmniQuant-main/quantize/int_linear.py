import torch
import torch.nn as nn
import torch.nn.functional as F
from quantize.quantizer import UniformAffineQuantizer


class QuantLinear(nn.Module):
    """
    Quantized Linear wrapper used by OmniQuant.

    Fix included:
      - F.linear requires input/weight/bias to share dtype.
      - In OmniQuant, LN/RMSNorm may output fp32 while weights are fp16 (or vice versa),
        especially under autocast / mixed precision and with temporary params.
      - We cast input to weight.dtype (and bias to weight.dtype) right before matmul.
    """

    def __init__(
        self,
        org_module: nn.Linear,
        weight_quant_params: dict = {},
        act_quant_params: dict = {},
        disable_input_quant: bool = False,
    ):
        super().__init__()
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear

        # Keep buffers (repo behavior) rather than Parameters
        self.register_buffer("weight", org_module.weight)
        if org_module.bias is not None:
            self.register_buffer("bias", org_module.bias)
        else:
            self.bias = None

        self.in_features = org_module.in_features
        self.out_features = org_module.out_features

        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False

        # initialize quantizers
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params, shape=org_module.weight.shape)
        if not disable_input_quant:
            self.act_quantizer = UniformAffineQuantizer(**act_quant_params)
        else:
            self.act_quantizer = None

        self.disable_input_quant = disable_input_quant
        self.use_temporary_parameter = False

    def forward(self, input: torch.Tensor):
        # Select weight/bias source
        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias
        elif self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.weight
            bias = self.bias

        # Optional activation quant
        if self.use_act_quant and not self.disable_input_quant:
            input = self.act_quantizer(input)

        # --- DTYPE FIX: make matmul operands consistent ---
        # F.linear will error if input.dtype != weight.dtype.
        # We cast input to weight dtype (common practice); keep device the same.
        if input.dtype != weight.dtype:
            input = input.to(dtype=weight.dtype)

        # Bias dtype must also match
        if bias is not None and bias.dtype != weight.dtype:
            bias = bias.to(dtype=weight.dtype)

        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
