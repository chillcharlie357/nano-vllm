# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Documentation Guidelines

### Time Stamps

**CRITICAL**: When creating or modifying markdown files in the `docs/` directory, always use the **current system time** for YAML front matter:

```bash
# Get current time before creating/modifying docs
date "+%Y-%m-%d %H:%M"
```

Example YAML front matter:
```yaml
---
title: Your Document Title
date: 2025-12-31 01:03    # ← Use current system time
modified: 2025-12-31 01:03 # ← Same as date for new files
tags:
  - tag1
  - tag2
categories:
  - 技术分享
excerpt: Brief description
mathjax: true
comment: true
---
```

## Common Commands

### Running Examples
```bash
python example.py    # Basic usage example
python bench.py      # Performance benchmark
```

### Installation
```bash
pip install -e .     # Development installation
```

### Model Download
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Architecture Overview

Nano-vLLM is a lightweight LLM inference engine (~1,200 lines) implementing vLLM-style PagedAttention with optimizations. The architecture consists of four interconnected subsystems:

### 1. Request Lifecycle (LLMEngine → Scheduler → ModelRunner)

The main entry point is `LLM` (nanovllm/llm.py:4), which inherits from `LLMEngine`:
- **LLMEngine.generate()**: High-level API that accepts prompts and runs the generation loop
- **LLMEngine.step()**: Executes one iteration of prefill or decode
  - Calls `scheduler.schedule()` to get batched sequences
  - Calls `model_runner.run()` to execute model forward pass
  - Calls `scheduler.postprocess()` to handle new tokens

### 2. Scheduler and Memory Management (engine/scheduler.py, engine/block_manager.py)

**Scheduler** implements a two-stage scheduling policy:
- **Prefill stage**: Admits waiting sequences if enough tokens and memory blocks available
- **Decode stage**: Runs ongoing sequences, preempting if needed (moves sequences back to waiting queue)
- Uses **BlockManager** for PagedAttention KV cache allocation

**BlockManager** implements prefix caching:
- Blocks are hash-based (xxhash) and can be shared across sequences
- `allocate()`: Assigns blocks to new sequences, reusing cached blocks when possible
- `may_append()`: Allocates new block when sequence crosses block boundary
- `deallocate()`: Frees blocks and decrements reference counts

### 3. Model Execution (engine/model_runner.py)

**ModelRunner** handles the actual model execution:
- Manages tensor parallelism via multiprocessing (spawn context, shared memory for RPC)
- **Warmup phase**: Runs max-length batch to measure peak memory and determine KV cache size
- **KV cache allocation**: Calculates available GPU memory and pre-allocates contiguous cache
- **Prefill vs Decode**:
  - Prefill: Uses Flash Attention varlen with cumulative sequence lengths
  - Decode: Uses Flash Attention with KV cache and block tables
- **CUDA Graph**: Captures graphs for decode at batch sizes [1, 2, 4, 8, 16, 32, ...] for reduced overhead
- Multi-GPU communication via NCCL and shared memory for method dispatch

### 4. Model Architecture (models/qwen3.py)

Currently only **Qwen3** is supported. The model structure:
- **Qwen3ForCausalLM**: Container with `Qwen3Model` + `lm_head`
- **Qwen3Model**: Embedding layers + decoder layers + final norm
- **Qwen3DecoderLayer**: Attention + MLP with pre-norm (RMSNorm)
- **Qwen3Attention**: QKV projection, rotary embedding, Flash Attention
  - Uses tensor-parallel linear layers (QKVParallelLinear, RowParallelLinear)
  - Q/K normalization when no bias
- **Qwen3MLP**: Gate-up projection (merged) + SiLU activation + down projection

### Key Implementation Details

**Prefix Caching**: Blocks compute hash of token IDs. If hash matches existing block, reuse cache. Sequence tracks `num_cached_tokens` to skip recomputation during prefill.

**PagedAttention**: KV cache divided into fixed-size blocks (default 256 tokens). Each sequence has a `block_table` mapping logical block indices to physical block IDs. Attention uses block tables to gather scattered KV cache.

**Context Management** (utils/context.py): Thread-local storage stores attention metadata (cu_seqlens, slot_mapping, block_tables) to avoid passing through all layers.

**Model Loading** (utils/loader.py): Loads HF checkpoints with weight name mapping via `packed_modules_mapping` (e.g., merges q/k/v projections into single qkv_proj).

**Attention** (layers/attention.py):
- Prefill: `flash_attn_varlen_func` with cumulative sequence lengths
- Decode: `flash_attn_with_kvcache` with block tables
- Custom Triton kernel `store_kvcache_kernel` writes KV to paged cache

**Sequence States** (engine/sequence.py):
- `WAITING`: In queue, not yet scheduled
- `RUNNING`: Currently being processed
- `FINISHED`: Completed (EOS or max_tokens reached)

## Adding Support for New Models

1. Create new model file in `nanovllm/models/` (e.g., `llama3.py`)
2. Implement model classes mirroring the Qwen3 structure
3. Define `packed_modules_mapping` to merge HF weights into nano-vLLM's fused layers
4. Import in `engine/model_runner.py` and update `ModelRunner.__init__` to load your model
5. Key requirements:
   - Use tensor-parallel linear layers from `layers/linear.py`
   - Use attention wrapper from `layers/attention.py`
   - Follow the `positions, hidden_states` forward signature for layers

## Configuration Options

Key config fields (nanovllm/config.py):
- `tensor_parallel_size`: Number of GPUs for model parallelism
- `enforce_eager`: Disable CUDA graph (useful for debugging)
- `max_model_len`: Maximum sequence length
- `gpu_memory_utilization`: Fraction of GPU memory to use (0-1)
- `max_num_seqs`: Maximum concurrent sequences
- `max_num_batched_tokens`: Maximum tokens in a batch
