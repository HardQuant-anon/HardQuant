import torch
import torch.nn as nn

from datasets import load_dataset
import functools
from collections import defaultdict

from functools import partial
import numpy as np
from tqdm import tqdm


def _unwrap_tensor(x):
    # Some hooks / model code pass (tensor,) instead of tensor
    while isinstance(x, (tuple, list)) and len(x) > 0:
        if torch.is_tensor(x[0]) and len(x) == 1:
            x = x[0]
            break
        if torch.is_tensor(x[0]) and len(x) >= 1:
            x = x[0]
            break
        x = x[0]
    return x


def _safe_stat_max_per_channel(t: torch.Tensor) -> torch.Tensor:
    """
    Return per-channel max over dim=0 after flattening to [N, H].
    If N==0, return an empty tensor on CPU (caller should skip).
    """
    if t is None or (not torch.is_tensor(t)):
        return torch.empty(0)

    if t.numel() == 0:
        return torch.empty(0)

    if t.dim() == 0:
        # scalar -> treat as H=1 with N=1
        t = t.view(1, 1)

    hidden_dim = int(t.shape[-1])
    if hidden_dim <= 0:
        return torch.empty(0)

    t2 = t.view(-1, hidden_dim)
    if t2.size(0) == 0:
        return torch.empty(0)

    return torch.max(t2, dim=0)[0].float().cpu()


def _tokenize_nonempty(tokenizer, text: str, seq_len: int):
    """
    Tokenize and return input_ids Tensor of shape [1, L] with L>0, or None if empty.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return None

    enc = tokenizer(
        text,
        return_tensors="pt",
        max_length=int(seq_len),
        truncation=True,
        add_special_tokens=True,  # helps ensure non-empty for some tokenizers
    )
    ids = enc.input_ids
    if not torch.is_tensor(ids) or ids.ndim != 2 or ids.size(0) != 1:
        return None
    if ids.size(1) == 0:
        return None
    return ids


def get_act_scales(model, tokenizer, dataset_path, num_samples=512, seq_len=512):
    """
    Robust to empty strings / empty tokenization results.
    Also robust to hooks receiving (tensor,) or empty tensors.
    """
    model.eval()
    device = next(model.parameters()).device
    act_scales = {}

    def stat_tensor(name, tensor):
        tensor = _unwrap_tensor(tensor)
        if tensor is None or (not torch.is_tensor(tensor)):
            return

        # Flatten and take abs max per hidden dimension
        t = tensor.detach().abs()
        coming_max = _safe_stat_max_per_channel(t)
        if coming_max.numel() == 0:
            return

        if name in act_scales:
            # both on CPU
            act_scales[name] = torch.max(act_scales[name], coming_max)
        else:
            act_scales[name] = coming_max

    def stat_input_hook(_m, x, _y, name):
        x = _unwrap_tensor(x)
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(functools.partial(stat_input_hook, name=name)))

    dataset = load_dataset("json", data_files=dataset_path, split="train").shuffle(seed=42)

    # Collect exactly num_samples *valid* samples (skip empties)
    n_valid = 0
    i = 0
    pbar = tqdm(total=int(num_samples), desc="Collecting act scales", leave=True)

    # Avoid infinite loops if the dataset is pathological
    max_tries = max(int(num_samples) * 50, 1000)

    while n_valid < int(num_samples) and i < len(dataset) and i < max_tries:
        txt = dataset[i].get("text", "")
        input_ids = _tokenize_nonempty(tokenizer, txt, int(seq_len))
        i += 1
        if input_ids is None:
            continue

        model(input_ids.to(device, non_blocking=True))
        n_valid += 1
        pbar.update(1)

    pbar.close()

    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass

    if n_valid < int(num_samples):
        raise RuntimeError(
            f"Only collected {n_valid}/{int(num_samples)} valid calibration samples "
            f"(skipped empties/zero-length). Consider a different dataset or reduce --num-samples."
        )

    return act_scales


@torch.no_grad()
def get_static_decoder_layer_scales(
    model,
    tokenizer,
    dataset_path,
    num_samples=512,
    seq_len=512,
):
    """
    Robust to empty strings / empty tokenization results.
    """
    model.eval()
    device = next(model.parameters()).device

    act_dict = defaultdict(dict)

    def stat_io_hook(_m, x, y, name):
        x = _unwrap_tensor(x)
        y = _unwrap_tensor(y)

        if torch.is_tensor(x) and x.numel() > 0:
            xin = float(x.detach().abs().max().item())
            if "input" not in act_dict[name]:
                act_dict[name]["input"] = xin
            else:
                act_dict[name]["input"] = max(act_dict[name]["input"], xin)

        if torch.is_tensor(y) and y.numel() > 0:
            yout = float(y.detach().abs().max().item())
            if "output" not in act_dict[name]:
                act_dict[name]["output"] = yout
            else:
                act_dict[name]["output"] = max(act_dict[name]["output"], yout)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(partial(stat_io_hook, name=name)))

    dataset = load_dataset("json", data_files=dataset_path, split="train").shuffle(seed=42)

    print("Collecting activation scales...")
    pbar = tqdm(total=int(num_samples), desc="Collecting layer scales", leave=True)

    n_valid = 0
    i = 0
    max_tries = max(int(num_samples) * 50, 1000)

    while n_valid < int(num_samples) and i < len(dataset) and i < max_tries:
        txt = dataset[i].get("text", "")
        input_ids = _tokenize_nonempty(tokenizer, txt, int(seq_len))
        i += 1
        if input_ids is None:
            continue

        model(input_ids.to(device, non_blocking=True))
        n_valid += 1

        # Only compute mean if we have any entries with "input"
        inputs = [v["input"] for v in act_dict.values() if isinstance(v, dict) and "input" in v]
        if inputs:
            mean_scale = float(np.mean(inputs))
            pbar.set_description(f"Mean input scale: {mean_scale:.2f}")

        pbar.update(1)

    pbar.close()

    for hook in hooks:
        try:
            hook.remove()
        except Exception:
            pass

    if n_valid < int(num_samples):
        raise RuntimeError(
            f"Only collected {n_valid}/{int(num_samples)} valid calibration samples "
            f"(skipped empties/zero-length). Consider a different dataset or reduce --num-samples."
        )

    # NOTE: This block is OPT-specific in the original code (model.decoder.layers.*)
    # If you run non-OPT models (e.g., Qwen/Llama), these keys will not exist.
    decoder_layer_scales = []
    for idx in range(int(model.config.num_hidden_layers)):
        scale_dict = {}

        def _req(key: str) -> float:
            if key not in act_dict or ("input" not in act_dict[key] and "output" not in act_dict[key]):
                raise KeyError(
                    f"Missing activation stats for '{key}'. "
                    "This function assumes an OPT-style module naming (model.decoder.layers.*)."
                )
            return act_dict[key]

        # Keep your original keys; raise clearly if model is not OPT-style
        scale_dict["attn_input_scale"] = act_dict[f"model.decoder.layers.{idx}.self_attn.q_proj"]["input"] / 127
        scale_dict["q_output_scale"] = act_dict[f"model.decoder.layers.{idx}.self_attn.q_proj"]["output"] / 127
        scale_dict["k_output_scale"] = act_dict[f"model.decoder.layers.{idx}.self_attn.k_proj"]["output"] / 127
        scale_dict["v_output_scale"] = act_dict[f"model.decoder.layers.{idx}.self_attn.v_proj"]["output"] / 127
        scale_dict["out_input_scale"] = act_dict[f"model.decoder.layers.{idx}.self_attn.out_proj"]["input"] / 127
        scale_dict["fc1_input_scale"] = act_dict[f"model.decoder.layers.{idx}.fc1"]["input"] / 127
        scale_dict["fc2_input_scale"] = act_dict[f"model.decoder.layers.{idx}.fc2"]["input"] / 127

        decoder_layer_scales.append(scale_dict)

    return decoder_layer_scales, act_dict
