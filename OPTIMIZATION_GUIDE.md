# llm.c Project Optimization Guide


## How the llm.c Repository Is Organized

### Top Level
- **train_gpt2_fp32.cu, train_gpt2.cu, train_gpt2.c**: Main training entry points. The train_gpt2_fp32.cu version is a simple, FP32-only CUDA implementation. train_gpt2.cu is the advanced, mainline CUDA version (supports mixed precision, multi-GPU, etc.). train_gpt2.c is a CPU-only reference.
 - **test_gpt2_fp32.cu, test_gpt2.cu, test_gpt2.c**: Correctness/regression harnesses (fixed data; forward/backward comparisons). Use these for validation or microbenchmarks, not for user-facing inference latency or text generation.
- **profile_gpt2.cu**: Profiling code for performance analysis of the **train_gpt2.cu** (mainline) training step.
- **train_llama3.py, train_gpt2.py**: Python scripts for orchestrating training or data preparation.
- **Makefile, requirements.txt**: Build and dependency management.

### dev/cuda/
- **matmul_forward.cu, matmul_backward.cu**: Matrix multiplication (GEMM) CUDA kernels.
- **attention_forward.cu, attention_backward.cu**: Attention mechanism CUDA kernels.
- **layernorm_forward.cu, gelu_forward.cu, softmax_forward.cu**: Other core CUDA kernels for transformer operations.
- **common.h**: Shared CUDA utilities and definitions.

> Note (important for this project scope): the kernels in **dev/cuda/** are primarily used by the **train_gpt2.cu** mainline code path (via the **llmc/** headers). If you are only optimizing kernels that are executed by **train_gpt2_fp32.cu**, you generally do **not** need to edit **dev/cuda/**.

### llmc/
- **.cuh/.h/.cpp files**: Core C++/CUDA headers and implementations for model components (attention, matmul, layernorm, etc.).

### dev/data/
- Data preparation and loading scripts (e.g., tinyshakespeare.py).

### dev/eval/
- Evaluation and export scripts.

### scripts/
- Shell scripts for running experiments and multi-node jobs.

----------

## Dataset and Model Weights

- The repo is typically trained and demoed on the TinyShakespeare dataset (a collection of Shakespeare’s works).
- The starter pack script (`dev/download_starter_pack.sh`) downloads:
   - The tokenized TinyShakespeare dataset (for training or fine-tuning).
   - The GPT-2 124M model weights, which are pre-trained weights from the original GPT-2 model (trained on a large corpus by OpenAI).
- This means you can use a pre-trained model for inference or further fine-tune it on TinyShakespeare for experiments.
- You can also generate the dataset yourself by running `python dev/data/tinyshakespeare.py`. This script downloads the raw Shakespeare text, tokenizes and formats it, and saves it in the format needed for training the model—so you don’t have to rely on the preprocessed version from the starter pack.


----------

## Profiling Notes (FP32-only vs Mainline)

- **`profile_gpt2.cu` and `profile_gpt2cu.py` profile `train_gpt2.cu`**, not `train_gpt2_fp32.cu`.
   - `profile_gpt2.cu` literally does `#include "train_gpt2.cu"` and runs a single training step for profiling.
- If you are optimizing **train_gpt2_fp32.cu-only kernels**, profile the `train_gpt2fp32cu` executable directly (Nsight Systems / Nsight Compute), or create an fp32-specific profiling wrapper analogous to `profile_gpt2.cu` that includes `train_gpt2_fp32.cu`.


----------

## Project Goal

Optimize the main bottlenecks in `train_gpt2_fp32.cu` (FP32-only path) and quantify improvements against a baseline.


## Key Files

- `llm.c/train_gpt2_fp32.cu`: training loop + FP32-path CUDA kernels.
- `llm.c/test_gpt2_fp32.cu`: correctness/regression test (logits/loss/gradients vs reference state).


## Metrics (concise)

Use a small set of metrics that directly supports your performance claims:

1. **Step time (ms/step)** at fixed `(B,T)`
2. **Throughput (tokens/s)** where `tokens/s = (B*T) / (seconds per step)`
3. **Top kernel time share (%)** (Nsight: which kernels dominate GPU time)
4. **Correctness gate (pass/fail)**: `test_gpt2fp32cu` must pass after each change

(Optional, only if relevant): peak VRAM / allocation size.


## Recommended Workflow (bottleneck-driven)

1. **Set up data + weights**
   - Run: `bash dev/download_starter_pack.sh`
   (Downloads TinyShakespeare dataset and GPT-2 124M weights).

2. **Build**
   - `make train_gpt2fp32cu`
   - `make test_gpt2fp32cu`

3. **Choose a fixed benchmark config**
   - Fix `(B,T)` and any other flags you will use for all comparisons.
   - Avoid frequent sampling/generation during timing runs (printing and per-token GPU->CPU copies can dominate and add noise).

4. **Baseline**
   - Warm up, then measure average ms/step over a window (e.g., 50-200 steps).
   - Record ms/step and tokens/s.

5. **Find bottlenecks**
   - Nsight Systems: identify whether time is in kernels vs memcpy vs CPU.
   - Nsight Compute: deep-dive the top 1-3 kernels.

6. **Optimize (one change at a time)**
   - Modify a single kernel or single data/memory decision.
   - Rebuild.

7. **Validate + compare**
   - Run `./test_gpt2fp32cu`.
   - Re-run the exact same benchmark config and compare to baseline using the metrics above.

8. **Document**
   - For each optimization: what changed, why it should help, and before/after metrics.