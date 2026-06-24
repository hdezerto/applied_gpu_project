# Accelerating `llm.c` GPT-2 with cuBLAS and Mixed Precision

This repository contains a DD2360 Applied GPU Programming project built on top of the `llm.c` GPT-2 training codebase. The project profiles and accelerates a CUDA implementation of GPT-2 by replacing custom linear-layer GEMMs with cuBLAS and adding an optional FP16 mixed-precision path for Tensor Core execution.

**Group members:**
- Diogo Paulo
- Hugo Dezerto
- Maria Carolina Sebastião

## Project Overview

The main project work is concentrated in:

- `train_gpt2_fp32.cu`: CUDA GPT-2 training implementation using FP32 master parameters and activations, with optional cuBLAS linear layers and mixed-precision Tensor Core matmul paths controlled by environment variables.
- `test_gpt2_fp32.cu`: correctness and performance test harness for the CUDA implementation, including forward-only inference timing.

The remaining directories contain support code, development kernels, dataset helpers, and scripts inherited from the `llm.c` project structure.

## Highlights

- CUDA training-loop benchmark with forward pass, loss, backward pass, and AdamW update.
- Profiling-driven optimization of the dominant custom matrix multiplication kernel.
- Optional cuBLAS-backed linear layers.
- Optional FP16 shadow-weight path for Tensor Core GEMMs while keeping master weights, gradients, optimizer state, and non-GEMM operations in FP32.
- Correctness checks against reference GPT-2 debug-state files.
- Training and inference throughput comparisons across optimization settings.
- Runtime sequence-length override for inference scaling tests.
- Starter-pack download script for model, tokenizer, and Tiny Shakespeare data.

## Results Summary

Experiments were run on a single NVIDIA Tesla T4 GPU using the GPT-2 124M checkpoint and Tiny Shakespeare data.

| Configuration | Training avg. iteration time | Training throughput | Inference throughput |
| --- | ---: | ---: | ---: |
| FP32 baseline | 1063.97 ms/iter | ~3,850 tokens/s | ~8,891 tokens/s |
| cuBLAS FP32 linear layers | 1118.69 ms/iter | ~3,660 tokens/s | ~11,946 tokens/s |
| cuBLAS + FP16 mixed precision | 834.44 ms/iter | ~4,910 tokens/s | ~26,940 tokens/s |

The custom FP32 matrix multiplication kernel dominated the baseline GPU runtime, taking about 58.65% of total GPU kernel time. cuBLAS alone improved inference throughput but slightly slowed the full training loop for this workload. The mixed-precision cuBLAS path improved training throughput by about 27.5% and gave more than a 3x inference throughput improvement over the FP32 baseline.

For forward-only inference scaling with batch size `B = 4`, the mixed-precision path reached up to a 5.66x speedup at sequence length `T = 64` and still achieved a 2.67x speedup at `T = 1024`.

## Repository Contents

| Path | Purpose |
| --- | --- |
| `train_gpt2_fp32.cu` | Main FP32 CUDA GPT-2 training program. |
| `test_gpt2_fp32.cu` | Test harness for correctness and timing checks. |
| `Makefile` | Build targets for the FP32 CUDA programs and available inherited targets. |
| `llmc/` | Shared headers and utilities from the `llm.c` codebase. |
| `dev/cuda/` | Development and benchmarking kernels. |
| `dev/data/` | Dataset preparation scripts. |
| `dev/download_starter_pack.sh` | Downloads the model, tokenizer, debug state, and small datasets needed for local runs. |
| `scripts/` | Example training scripts for larger GPT runs. |
| `doc/` | Supporting technical notes. |

## Requirements

The experiments were run in Google Colab on an NVIDIA Tesla T4 GPU. The project is intended for a Linux CUDA environment with an NVIDIA GPU.

Required:

- CUDA toolkit with `nvcc`
- cuBLAS / cuBLASLt
- C/C++ compiler supported by CUDA
- `make`

Optional, depending on which inherited targets or scripts you run:

- OpenMP
- NCCL
- MPI
- cuDNN
- Python packages used by the dataset and evaluation helpers

## Quickstart

Download the starter files:

```bash
bash dev/download_starter_pack.sh
```

Build the project-specific CUDA programs:

```bash
make train_gpt2fp32cu test_gpt2fp32cu
```

Run the correctness and timing test:

```bash
./test_gpt2fp32cu
```

Run a small training smoke test:

```bash
./train_gpt2fp32cu -b 4 -t 64 -v 20 -m 5 -s 20 -g 32
```

The default data paths point to the Tiny Shakespeare files downloaded by `dev/download_starter_pack.sh`.

## Runtime Options

The FP32 training program supports command-line options for dataset paths, batch size, sequence length, learning rate, validation frequency, and sample generation length:

```bash
./train_gpt2fp32cu -i <train_data.bin> -j <val_data.bin> -b 4 -t 1024 -l 3e-4
```

Useful environment variables:

```bash
# Use cuBLAS for linear layers instead of the custom matmul kernel.
LLMC_ENABLE_CUBLAS_LINEAR=1 ./train_gpt2fp32cu

# Use FP16 shadow weights for Tensor Core GEMMs where supported.
LLMC_ENABLE_MIXED_PRECISION=1 LLMC_ENABLE_CUBLAS_LINEAR=1 ./train_gpt2fp32cu

# Override sequence length in the test harness for scaling experiments.
OVERRIDE_T=128 ./test_gpt2fp32cu
```

## Project Status

This repository is prepared for source-code review and portfolio presentation. Generated binaries, downloaded model files, dataset shards, logs, and benchmark outputs are intentionally ignored by Git.

## Credits

This project builds on the public `llm.c` codebase and was developed for DD2360 Applied GPU Programming at KTH Royal Institute of Technology.
