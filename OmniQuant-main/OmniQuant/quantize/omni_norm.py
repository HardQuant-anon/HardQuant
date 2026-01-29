import torch
import torch.nn as nn


"""
Modify normalization layer to adapt the training of learnable equivalent transformation
"""


def _unwrap_tensor(x):
    """
    OmniQuant sometimes gets passed hidden_states as (tensor,) or (tensor, ...).
    Transformers expects tensor hidden_states; be permissive here.
    """
    if torch.is_tensor(x):
        return x
    if isinstance(x, (tuple, list)) and len(x) > 0:
        # Common cases: (tensor,) or (tensor, past_kv/attn/etc)
        if torch.is_tensor(x[0]):
            return x[0]
        # Peel nested singletons: ((tensor,),) etc.
        y = x
        while isinstance(y, (tuple, list)) and len(y) == 1:
            y = y[0]
            if torch.is_tensor(y):
                return y
            if isinstance(y, (tuple, list)) and len(y) > 0 and torch.is_tensor(y[0]):
                return y[0]
    return x


class OmniLayerNorm(nn.Module):
    def __init__(self, ori_layer_norm) -> None:
        super().__init__()
        self.use_act_quant = True
        self.register_buffer("weight", ori_layer_norm.weight)
        if ori_layer_norm.bias is not None:
            self.register_buffer("bias", ori_layer_norm.bias)
        else:
            self.bias = None
        self.eps = ori_layer_norm.eps
        self.norm_func = nn.functional.layer_norm
        self.normalized_shape = ori_layer_norm.normalized_shape
        self.use_temporary_parameter = False

    def forward(self, x):
        x = _unwrap_tensor(x)
        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias
        else:
            weight = self.weight
            bias = self.bias

        out = self.norm_func(x, self.normalized_shape, weight, bias, eps=self.eps)
        return out

    def set_quant_state(self, use_weight_quant, use_act_quant):
        self.use_act_quant = use_act_quant


class OmniLlamaRMSNorm(nn.Module):
    def __init__(self, ori_norm, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.register_buffer("weight", ori_norm.weight)
        self.bias = None
        self.variance_epsilon = eps
        self.use_temporary_parameter = False

    def forward(self, hidden_states):
        hidden_states = _unwrap_tensor(hidden_states)

        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias
        else:
            weight = self.weight
            bias = self.bias if hasattr(self, "bias") else None

        if bias is not None:
            return (weight * hidden_states + bias).to(input_dtype)
        return (weight * hidden_states).to(input_dtype)
