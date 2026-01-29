from collections import OrderedDict

import torch
import torch.nn as nn

from quantize.int_linear import QuantLinear
from quantize.int_matmul import QuantMatMul
from models.transformation import *


def let_parameters(model, use_shift=True):
    params = []
    template = "smooth" if use_shift else "smooth_scale"
    for n, m in model.named_parameters():
        if n.find(template) > -1:
            params.append(m)
    return iter(params)


def lwc_parameters(model):
    params = []
    for n, m in model.named_parameters():
        if n.find("bound_factor") > -1:
            params.append(m)
    return iter(params)


def get_omni_parameters(model, use_shift=True):
    params = []
    template = "smooth" if use_shift else "smooth_scale"
    for n, m in model.named_parameters():
        if n.find("bound_factor") > -1 or n.find(template) > -1:
            params.append(m)
    return iter(params)


def omni_state_dict(model, destination=None, prefix="", keep_vars=False):
    if destination is None:
        destination = OrderedDict()
    for name, param in model.named_parameters():
        if name.find("smooth") > -1 or name.find("bound_factor") > -1:
            destination[prefix + name] = param if keep_vars else param.detach()
    return destination


def register_scales_and_zeros(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.weight_quantizer.register_scales_and_zeros()


class TruncateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        truncated_tensor = input.clone()
        m = truncated_tensor.abs() < threshold
        truncated_tensor[m] = truncated_tensor[m].sign() * threshold
        return truncated_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None


def truncate_number(number, threshold=1e-2):
    # avoid overflow with AMP training
    return TruncateFunction.apply(number, threshold)


def _safe_set_temp(module: nn.Module, name: str, value: torch.Tensor) -> None:
    """
    Fixes: TypeError cannot assign Tensor as parameter 'temp_weight'.

    If module.<name> is a registered nn.Parameter, we must update it in-place
    (or replace with an nn.Parameter). If it is not registered, we can attach
    a plain Tensor attribute.
    """
    value = value.detach()
    cur = getattr(module, name, None)

    # If it's already a Parameter: update .data (or replace if shape changed)
    if isinstance(cur, nn.Parameter):
        if cur.data.shape != value.shape:
            setattr(module, name, nn.Parameter(value, requires_grad=False))
        else:
            cur.data.copy_(value)
        return

    # If it is registered in _parameters (even if None), assignment must be nn.Parameter or None
    if hasattr(module, "_parameters") and isinstance(module._parameters, dict) and name in module._parameters:
        setattr(module, name, nn.Parameter(value, requires_grad=False))
        return

    # Otherwise it's a normal attribute
    setattr(module, name, value)


def smooth_and_quant_temporary(model, args, isllama):
    if args.let:
        with torch.no_grad():
            for name, module in model.named_parameters():
                if "smooth_scale" in name:
                    module.data = truncate_number(module)

        if isllama:
            smooth_ln_fcs_temporary(
                model.input_layernorm,
                [model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
                model.qkv_smooth_scale,
                model.qkv_smooth_shift,
            )
            smooth_ln_fcs_temporary(
                model.post_attention_layernorm,
                [model.mlp.up_proj, model.mlp.gate_proj],
                model.fc1_smooth_scale,
                model.fc1_smooth_shift,
            )
            smooth_fc_fc_temporary(
                model.self_attn.v_proj,
                model.self_attn.o_proj,
                model.out_smooth_scale,
                model.out_smooth_shift,
            )
            smooth_q_k_temporary(
                model.self_attn.q_proj,
                model.self_attn.k_proj,
                model.qkt_smooth_scale,
            )

            # TEMP FIX: don't assign Tensor into a registered Parameter slot
            _safe_set_temp(model.mlp.down_proj, "temp_weight", model.mlp.down_proj.weight)

        else:
            smooth_ln_fcs_temporary(
                model.self_attn_layer_norm,
                [model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
                model.qkv_smooth_scale,
                model.qkv_smooth_shift,
            )
            smooth_ln_fcs_temporary(
                model.final_layer_norm,
                [model.fc1],
                model.fc1_smooth_scale,
                model.fc1_smooth_shift,
            )
            smooth_ln_fcs_temporary(
                model.self_attn.v_proj,
                model.self_attn.out_proj,
                model.out_smooth_scale,
                model.out_smooth_shift,
            )
            smooth_q_k_temporary(
                model.self_attn.q_proj,
                model.self_attn.k_proj,
                model.qkt_smooth_scale,
            )

            # TEMP FIX: don't assign Tensor into a registered Parameter slot
            _safe_set_temp(model.fc2, "temp_weight", model.fc2.weight)

    else:
        # Ensure every QuantLinear has a temp_weight pointing at weight (safely)
        for name, module in model.named_modules():
            if isinstance(module, QuantLinear):
                _safe_set_temp(module, "temp_weight", module.weight)

    # quant
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            # Ensure temp_weight exists (safely)
            if hasattr(module, "temp_weight"):
                tw = module.temp_weight
            else:
                tw = module.weight
                _safe_set_temp(module, "temp_weight", tw)

            # Quantize temp_weight and write back safely
            qtw = module.weight_quantizer(tw)
            _safe_set_temp(module, "temp_weight", qtw)

            # Bias: keep temp_bias as a plain attribute (or set safely if it happens to be registered)
            if not hasattr(module, "temp_bias"):
                # some modules may have bias=None
                tb = module.bias
                _safe_set_temp(module, "temp_bias", tb) if tb is not None else setattr(module, "temp_bias", None)

            module.use_temporary_parameter = True


def clear_temp_variable(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            # If temp_* are registered parameters, they must be set to None (not deleted)
            if hasattr(module, "_parameters") and isinstance(module._parameters, dict):
                if "temp_weight" in module._parameters:
                    setattr(module, "temp_weight", None)
                elif hasattr(module, "temp_weight"):
                    delattr(module, "temp_weight")

                if "temp_bias" in module._parameters:
                    setattr(module, "temp_bias", None)
                elif hasattr(module, "temp_bias"):
                    delattr(module, "temp_bias")
            else:
                if hasattr(module, "temp_weight"):
                    delattr(module, "temp_weight")
                if hasattr(module, "temp_bias"):
                    delattr(module, "temp_bias")


@torch.no_grad()
def smooth_and_quant_inplace(model, args, isllama):
    if args.let:
        for name, module in model.named_parameters():
            if "smooth_scale" in name:
                module.data = truncate_number(module)
        if isllama:
            smooth_ln_fcs_inplace(
                model.input_layernorm,
                [model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
                model.qkv_smooth_scale,
                model.qkv_smooth_shift,
            )
            smooth_ln_fcs_inplace(
                model.post_attention_layernorm,
                [model.mlp.up_proj, model.mlp.gate_proj],
                model.fc1_smooth_scale,
                model.fc1_smooth_shift,
            )
            smooth_fc_fc_inplace(
                model.self_attn.v_proj,
                model.self_attn.o_proj,
                model.out_smooth_scale,
                model.out_smooth_shift,
            )
        else:  # opt
            smooth_ln_fcs_inplace(
                model.self_attn_layer_norm,
                [model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
                model.qkv_smooth_scale,
                model.qkv_smooth_shift,
            )
            smooth_ln_fcs_inplace(
                model.final_layer_norm,
                [model.fc1],
                model.fc1_smooth_scale,
                model.fc1_smooth_shift,
            )
            smooth_fc_fc_inplace(
                model.self_attn.v_proj,
                model.self_attn.out_proj,
                model.out_smooth_scale,
                model.out_smooth_shift,
            )
        smooth_q_k_inplace(model.self_attn.q_proj, model.self_attn.k_proj, model.qkt_smooth_scale)

    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.weight = module.weight_quantizer(module.weight)
            module.use_temporary_parameter = False


def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
    # setting weight quantization here does not affect actual forward pass
    self.use_weight_quant = weight_quant
    self.use_act_quant = act_quant
    for m in self.modules():
        if isinstance(m, (QuantLinear, QuantMatMul)):
            m.set_quant_state(weight_quant, act_quant)
