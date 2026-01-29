# llama.py
import time
import inspect

import torch
import torch.nn as nn

from gptq import *
from modelutils import *
from quant import *


# -----------------------------------------------------------------------------
# Device helpers (robust across Transformers rotary embedding implementations)
# -----------------------------------------------------------------------------
def _force_move_rotary_emb_to(rotary_emb: nn.Module, device):
    """
    Some Transformers versions keep rotary-embedding internals (e.g., inv_freq / caches)
    on CPU even if the module is moved. Force-move common tensors + all buffers.
    """
    if rotary_emb is None:
        return

    # Move the module parameters/buffers via .to()
    rotary_emb.to(device)

    # Force-move *all* registered buffers (covers many versions)
    for k, v in list(rotary_emb._buffers.items()):
        if torch.is_tensor(v):
            rotary_emb._buffers[k] = v.to(device)

    # Also handle common non-buffer attributes seen in different releases
    for attr in [
        "inv_freq",
        "inv_freq_expanded",
        "cos_cached",
        "sin_cached",
        "_cos_cached",
        "_sin_cached",
        "cos",
        "sin",
    ]:
        if hasattr(rotary_emb, attr):
            t = getattr(rotary_emb, attr)
            if torch.is_tensor(t):
                setattr(rotary_emb, attr, t.to(device))


def _get_model_rotary_emb(model: nn.Module):
    # LlamaForCausalLM -> model.model is LlamaModel
    m = getattr(model, "model", None)
    if m is None:
        return None
    return getattr(m, "rotary_emb", None)


def _ensure_model_rotary_on_device(model: nn.Module, device):
    rotary = _get_model_rotary_emb(model)
    if rotary is not None:
        _force_move_rotary_emb_to(rotary, device)


def _decoder_layer_forward_with_optional_position_embeddings(
    model: nn.Module,
    layer: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
):
    """
    Transformers has changed Llama attention APIs over time.
    Newer versions often require `position_embeddings=(cos, sin)` to be passed into the layer.
    Older versions ignore it.

    This wrapper checks the layer.forward signature and supplies position_embeddings when needed.
    """
    kwargs = dict(attention_mask=attention_mask, position_ids=position_ids)

    try:
        sig = inspect.signature(layer.forward)
        needs_pos_emb = "position_embeddings" in sig.parameters
    except Exception:
        needs_pos_emb = False

    if needs_pos_emb:
        rotary = _get_model_rotary_emb(model)
        if rotary is None:
            # Best effort fallback: some versions put rotary_emb on self_attn
            rotary = getattr(getattr(layer, "self_attn", None), "rotary_emb", None)

        if rotary is not None:
            _force_move_rotary_emb_to(rotary, hidden_states.device)
            pos_emb = rotary(hidden_states, position_ids)
            # Only pass if we actually have it; passing None can crash.
            kwargs["position_embeddings"] = pos_emb

    return layer(hidden_states, **kwargs)[0]


def get_llama(model_name_or_path: str):
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    from transformers import LlamaForCausalLM

    model = LlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype="auto")
    model.seqlen = 2048
    return model


@torch.no_grad()
def llama_sequential(model, dataloader, dev):
    print("Starting ...")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # Move the minimal components needed for the catcher pass to GPU
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    _ensure_model_rotary_on_device(model, dev)  # IMPORTANT for newer Transformers
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])

    for batch in dataloader:
        try:
            # Make sure rotary stays on the same device as the forward tensors
            _ensure_model_rotary_on_device(model, dev)
            model(batch[0].to(dev))
        except ValueError:
            pass

    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    _ensure_model_rotary_on_device(model, dev)

    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    print("Ready.")

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)

        if args.true_sequential:
            sequential = [
                ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
                ["self_attn.o_proj"],
                ["mlp.up_proj", "mlp.gate_proj"],
                ["mlp.down_proj"],
            ]
        else:
            sequential = [list(full.keys())]

        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = Quantizer()
                gptq[name].quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False)

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            for j in range(args.nsamples):
                outs[j] = _decoder_layer_forward_with_optional_position_embeddings(
                    model=model,
                    layer=layer,
                    hidden_states=inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )

            for h in handles:
                h.remove()

            for name in subset:
                print(i, name)
                print("Quantizing ...")
                gptq[name].fasterquant(
                    percdamp=args.percdamp,
                    groupsize=args.groupsize,
                    actorder=args.act_order,
                    static_groups=args.static_groups,
                )
                quantizers[f"model.layers.{i}.{name}"] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = _decoder_layer_forward_with_optional_position_embeddings(
                model=model,
                layer=layer,
                hidden_states=inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    return quantizers


@torch.no_grad()
def llama_eval(model, testenc, dev):
    print("Evaluating ...")

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    _ensure_model_rotary_on_device(model, dev)  # IMPORTANT for newer Transformers
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])

    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(dev)
        try:
            _ensure_model_rotary_on_device(model, dev)
            model(batch)
        except ValueError:
            pass

    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    _ensure_model_rotary_on_device(model, dev)  # keep on GPU for layer calls
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        if args.nearest:
            subset = find_layers(layer)
            for name in subset:
                quantizer = Quantizer()
                quantizer.configure(args.wbits, perchannel=True, sym=False, mse=False)
                W = subset[name].weight.data
                quantizer.find_params(W, weight=True)
                subset[name].weight.data = quantize(W, quantizer.scale, quantizer.zero, quantizer.maxq).to(
                    next(iter(layer.parameters())).dtype
                )

        for j in range(nsamples):
            outs[j] = _decoder_layer_forward_with_optional_position_embeddings(
                model=model,
                layer=layer,
                hidden_states=inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache


def llama_pack3(model, quantizers):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant3(model, quantizers)
    qlayers = find_layers(model, [Quant3Linear])
    print("Packing ...")
    for name in qlayers:
        print(name)
        quantizers[name] = quantizers[name].cpu()
        qlayers[name].pack(layers[name], quantizers[name].scale, quantizers[name].zero)
    print("Done.")
    return model


if __name__ == "__main__":
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model",
        type=str,
        help="LlaMa model to load; pass location of huggingface converted checkpoint.",
    )
    parser.add_argument(
        "dataset",
        type=str,
        choices=["wikitext103", "wikitext2", "ptb", "c4"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for sampling the calibration data.")
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration data samples.")
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument("--nearest", action="store_true", help="Whether to run the RTN baseline.")
    parser.add_argument(
        "--wbits",
        type=int,
        default=16,
        choices=[2, 3, 4, 8, 16],
        help="#bits to use for quantization; use 16 for evaluating base model.",
    )
    parser.add_argument(
        "--groupsize",
        type=int,
        default=-1,
        help="Groupsize to use for quantization; default uses full row.",
    )
    parser.add_argument("--sym", action="store_true", help="Whether to perform symmetric quantization.")
    parser.add_argument("--save", type=str, default="", help="Save quantized checkpoint under this name.")
    parser.add_argument("--new-eval", action="store_true", help="Whether to use the new PTB and C4 eval.")
    parser.add_argument("--act-order", action="store_true", help="Whether to apply the activation order GPTQ heuristic")
    parser.add_argument("--true-sequential", action="store_true", help="Whether to run in true sequential model.")
    parser.add_argument(
        "--static-groups",
        action="store_true",
        help="Whether to use static groups; recommended when using `--actorder` for more efficient inference.",
    )

    args = parser.parse_args()

    # DEV is usually defined in modelutils.py in this repo; add a safe fallback.
    try:
        DEV
    except NameError:
        DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = get_llama(args.model)
    model.eval()

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    if args.wbits < 16 and not args.nearest:
        tick = time.time()
        quantizers = llama_sequential(model, dataloader, DEV)
        print(time.time() - tick)

    if args.save:
        llama_pack3(model, quantizers)
        torch.save(model.state_dict(), args.save)
