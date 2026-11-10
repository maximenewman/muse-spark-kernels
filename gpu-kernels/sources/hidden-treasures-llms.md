---
title: SFU Hidden-Treasures / LLMs (seed source)
type: source
tags: [gpu, index, seed]
source_url: https://github.com/sfu-arch/Hidden-Treasures/tree/master/LLMs
local_copy: raw/hidden-treasures-llms/
---
# SFU Hidden-Treasures / LLMs (seed source)

**TL;DR**
- A teaching repo from SFU's architecture group (sfu-arch), written for first-year students. It's mostly a curated link list plus one real kernel project.
- The kernel-relevant parts are the **CUDA reading list** (Hopper GEMM, TMA, CuTe, PDL, smem banking, CUDA graphs, tensor cores) and **`3-GEMM/`**, a copy of alexarmbr's six-step tensor-core HGEMM (kernel 6 reaches about 95% of `cublasHgemm` at 8192³ on a T4).
- `swizzle.py` is a short demo of the Hopper XOR swizzle and how it maps addresses to banks.
- The LLM basics (GGUF, llama.cpp, vLLM, Unsloth/LoRA fine-tuning, `01_llm_tuning.py`, `02_llm_internals.ipynb`) don't help with kernel work, so this knowledge base skips them.

## What's in the repo
| Path | What | KB note |
|---|---|---|
| `README.md` | LLM primer + CUDA reading list | this note |
| `3-GEMM/src/kernel{1..6}.cu` | Progressive HGEMM: naive → smem tiling → … → ~95% cuBLAS | [[hidden-treasures-3-gemm-code]], [[alexarmbr-tensor-core-hgemm]] |
| `3-GEMM/profile.sh` | Locks GPU/DRAM clocks, then profiles all kernels against cuBLAS | [[benchmarking-kernels]] |
| `3-GEMM/scripts/shmem_layout_viz.py` | Visualises the shared-memory layout | [[swizzling]] |
| `swizzle.py` | XOR-swizzle bank mapping (8 banks × 16 B) | [[swizzling]], [[smem-bank-conflicts]] |
| `01_llm_tuning.py`, `02_llm_internals.ipynb`, `lora.png` | Unsloth fine-tuning, LLM internals | not covered |

## Reading list → KB notes
Hardware / architecture
- Modal GPU glossary (tensor core) → [[modal-gpu-glossary]]
- Fabien Sanglard, brief history of GPUs → [[sanglard-gpu-history]]
- SemiAnalysis, tensor-core evolution Volta→Blackwell → [[semianalysis-tensor-core-evolution]], [[tensor-core-generations]]
- GTC25 S72683 (memory model / max bandwidth & latency) → [[gtc25-s72683-memory]], [[memory-bandwidth-and-latency]]
- GTC25 S72449 → [[gtc25-s72449-modern-cuda]]. The repo labels it "Modern GPU Programming with CUDA", but the session is actually "1,001 Ways to Write CUDA Kernels in Python" (Leo Fang).

GEMM worklogs
- siboehm, CUDA SGEMM → [[siboehm-sgemm-worklog]]
- alexarmbr, tensor-core HGEMM from scratch → [[alexarmbr-tensor-core-hgemm]]
- cudaforfun, outperforming cuBLAS on H100 → [[h100-gemm-worklog-cudaforfun]]
- Colfax, Blackwell GEMM with tensor memory → [[colfax-blackwell-tmem-gemm]]
- Bruce-Lee-LY, WMMA / MMA PTX → [[bruce-lee-mma-ptx]], [[mma-ptx]]

Memory / layout
- veitner, fast matrix transpose on Hopper → [[veitner-hopper-transpose]]
- feldmann, smem microbenchmarks → [[feldmann-smem-microbenchmarks]], [[smem-bank-conflicts]]
- Yang YiFan blogs: TMA → [[yangyifan-cute-tma]] / [[tma]]; register tiling → [[yangyifan-register-tiling]] / [[register-tiling]]; CuTe layouts → [[yangyifan-cute-layouts]] / [[cute-layouts]]; PDL → [[yangyifan-pdl]] / [[pdl]]

Serving / DSLs
- PyTorch "Accelerating generative AI 2" (gpt-fast, CUDA graphs) → [[pytorch-gpt-fast-accelerating-genai-2]], [[cuda-graphs]]
- Triton → [[triton-docs]], [[triton]]
- Modular, Triton and Python eDSLs → [[modular-triton-and-python-edsls]]

## swizzle.py in one paragraph
`bank8_id(addr, s)` turns a byte address into a 4-byte word index, takes `s` bits from above the low 3 word bits (`(word >> 3) & ((1<<s)-1)`), XORs them into the word, and keeps the low 3 bits to get one of 8 bank groups. The idea: XORing row bits into column bits sends successive rows of a tile to different banks. The script only prints the mapping for s=0, and its plotting code is commented out. **Bug:** the docstring describes 16-byte slots and 128-byte rows, but `addr >> 2` makes the slots 4-byte words and the rows 32 bytes. Changing it to `addr >> 4` gives the intended model, and s=3 then reproduces Hopper's 128B swizzle (checked by re-implementing the logic, not by running the file; see [[swizzling]]). See [[swizzling]] for the full Hopper 32/64/128B swizzle modes.
