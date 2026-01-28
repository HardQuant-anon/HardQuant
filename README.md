# HardQuant (ICML 2026 submission code)

This repository contains the code corresponding to the **HardQuant** submission to **ICML 2026**.

At the top level, the workspace is organized as:

- `HardQuant/` — **main codebase** for HardQuant fine-tuning and experiments (core training pipelines live here)
- `smoothquant-compare/` — SmoothQuant baseline code
- `OmniQuant-main/` — OmniQuant baseline code
- `gptq-main/` — GPTQ baseline code
- `requirements.txt` — Python dependencies used across the workspace

## Quick start

### 1) Environment setup

Create an environment (conda/venv) and install dependencies:

```bash
python -m pip install -r requirements.txt
```

You will also need a working CUDA + PyTorch setup for GPU training.

### 2) Hugging Face access

Most runs assume Hugging Face model access (e.g., Llama-3 weights). Make sure you are logged in:

```bash
huggingface-cli login
```

(Alternatively set `HF_TOKEN` in your environment.)

## Fine-tuning (main entrypoint)

The primary fine-tuning pipeline is:

- `HardQuant/scripts/fine_tuning/core_alg/pipeline_distributed.py`

Runs below use `torchrun` for distributed execution (e.g., 2 GPUs on one node).

### Llama (Meta-Llama-3-8B)

```bash
torchrun --standalone --nproc_per_node=2     HardQuant/scripts/fine_tuning/core_alg/pipeline_distributed.py     --model meta-llama/Meta-Llama-3-8B     --dataset-in wikitext:wikitext-2-raw-v1     --dataset-unseen lambada:plain_text     --dataset-in-train-split train     --dataset-in-eval-split test     --dataset-unseen-eval-split validation     --steps 1000     --block-size 256     --batch-size-ft 1     --batch-size-eval 1     --lr 1e-6     --weight-decay 0.0     --lambda-x 1     --lambda-w 0.0     --lambda-factor 2.0     --weight-quant per_channel     --act-quant per_token     --weight-bits 4     --act-bits 4     --eval-every 10000     --log-every 10     --seeds 1     --gradient-checkpointing     --loss-type layer_wise     --layer-wise-k 80     --layer-wise-teacher-dtype fp16     --fsdp     --fsdp-sharding full_shard     --fsdp-mixed-precision bf16     --no-fsdp-use-orig-params
```

### Mistral (Mistral-7B-v0.1)

```bash
torchrun --standalone --nproc_per_node=2     HardQuant/scripts/fine_tuning/core_alg/pipeline_distributed.py     --model mistralai/Mistral-7B-v0.1     --dataset-in wikitext:wikitext-2-raw-v1     --dataset-unseen lambada:plain_text     --dataset-in-train-split train     --dataset-in-eval-split test     --dataset-unseen-eval-split validation     --steps 1000     --block-size 256     --batch-size-ft 1     --batch-size-eval 1     --lr 1e-6     --weight-decay 0.0     --lambda-x 1     --lambda-w 0.0     --weight-quant per_channel     --act-quant per_token     --weight-bits 4     --act-bits 4     --eval-every 10000     --log-every 10     --seeds 1     --gradient-checkpointing     --loss-type snows     --snows-k 80     --snows-teacher-dtype fp16     --fsdp     --fsdp-sharding full_shard     --fsdp-mixed-precision bf16     --no-fsdp-use-orig-params
```

## Evaluation / PTQ benchmarking

Evaluation is driven by:

- `HardQuant/scripts/evaluate/ptq_evaluate.py`

In all examples below, `--model-dir` can be a Hugging Face model id (e.g., `mistralai/Mistral-7B-v0.1`) or a local path to a checkpoint directory. This means you can evaluate PTQ either on the base model or on a **fine-tuned HardQuant checkpoint** by replacing:

- `--model-dir mistralai/Mistral-7B-v0.1`

with something like:

- `--model-dir /path/to/your/fine_tuned_checkpoint`

### SmoothQuant (example)

```bash
python3 HardQuant/scripts/evaluate/ptq_evaluate.py   --model-dir mistralai/Mistral-7B-v0.1   --device cuda:0   --eval-fraction 1   --block-size 256   --ptq smoothquant   --smoothquant-alpha 0.5   --smoothquant-calib-num-samples 1024   --smoothquant-calib-seq-len 256   --smoothquant-calib-dataset-name wikitext   --smoothquant-calib-dataset-split train   --smoothquant-device-map cuda:0   --weight-bits 4   --act-bits 4   --weight-quant per_channel   --act-quant per_token   --quantize-bmm-input   --quant-reload-instead-of-deepcopy
```

### GPTQ (example)

```bash
python3 HardQuant/scripts/evaluate/ptq_evaluate.py   --model-dir mistralai/Mistral-7B-v0.1   --device cuda:0   --eval-fraction 1   --ptq gptq_fakeact   --gptq-calib wikitext:wikitext-103-raw-v1::train   --gptq-nsamples 1024   --gptq-seqlen 256   --gptq-wbits 4   --gptq-groupsize -1   --gptq-percdamp 0.01   --gptq-true-sequential   --gptq-act-order   --gptq-sym   --act-quant per_token   --act-bits 4   --act-location input   --quant-reload-instead-of-deepcopy
```

### OmniQuant (example)

```bash
python3 HardQuant/scripts/evaluate/ptq_evaluate.py   --model-dir Qwen/Qwen3-14B   --device cuda:0   --ptq omniquant   --omniquant-dir /workspace/OmniQuant-main   --omniquant-net qwen3-14b   --omniquant-act-scales /workspace/OmniQuant-main/act_scales/qwen3-14b.pt   --omniquant-act-shifts /workspace/OmniQuant-main/act_shifts/qwen3-14b.pt   --omniquant-generate-missing-act-scales-shifts   --omniquant-seqlen 2048   --omniquant-calib-dataset wikitext103   --omniquant-nsamples 128   --omniquant-wbits 6   --omniquant-abits 6   --omniquant-a-dynamic-method per_token   --omniquant-w-dynamic-method per_channel   --omniquant-let   --omniquant-lwc   --omniquant-let-lr 1e-6   --omniquant-lwc-lr 1e-6   --omniquant-deactive-amp   --eval-fraction 1   --omniquant-net qwen3-14b   --omniquant-batch-size 8
```