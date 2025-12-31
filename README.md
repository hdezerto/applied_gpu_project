# applied_gpu_project

DD2360 project exploring reduced-precision inference and NVIDIA library acceleration for the `llm.c` GPT-style model.

## Repository Layout
- notebooks/DD2360_llm_c_setup.ipynb — Google Colab workflow that bootstraps the baseline FP32 CUDA pipeline (clone `llm.c`, build `test_gpt2fp32cu`, regenerate datasets, handle T4 arch flags).
- llm.c/ — local copy of Karpathy's CUDA GPT-2 implementation with our mixed-precision + cuBLAS enhancements (see below for usage).

## Notebook Workflow
1. Open the notebook in Colab or VS Code’s Jupyter editor.
2. Run the cells sequentially to reproduce the FP32 baseline build and data preparation.
3. Extend the notebook with reduced-precision + cuBLAS experiments or convert it into scripts as the project evolves.

## llm.c Mixed-Precision Build
The copied `llm.c/` tree already contains the FP16/cuBLAS changes. To rebuild and test on a T4 (compute capability 7.5):

```bash
cd llm.c
LLMC_ENABLE_MIXED_PRECISION=1 make test_gpt2fp32cu GPU_COMPUTE_CAPABILITY=75
LLMC_ENABLE_MIXED_PRECISION=1 ./test_gpt2fp32cu
```

- Unset the environment variable (or set it to `0`) to fall back to the original FP32 path for accuracy baselines.
- The binary prints whether mixed precision is enabled so you can capture both runtime and accuracy deltas easily.