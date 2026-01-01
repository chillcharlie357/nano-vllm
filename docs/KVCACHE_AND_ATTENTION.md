---
title: KV Cache 内存管理与多头注意力机制
date: 2025-12-31 01:03
tags:
  - vllm
  - AIInfra
  - LLM
  - 推理引擎
categories:
  - 技术分享
excerpt: 深入理解 LLM 推理引擎的 KV Cache 内存计算和多头注意力架构
mathjax: true
comment: true
---

# KV Cache 内存管理与多头注意力机制

> 深入理解 Nano-vLLM 的内存分配策略和注意力架构设计

本文档详细解答两个核心问题：
1. KV Cache 的内存是如何计算的？
2. KV head 和 Attention head 有什么区别？

---

# 目录

1. [KV Cache 内存计算详解](#kv-cache-内存计算详解)
2. [多头注意力架构对比](#多头注意力架构对比)

---

# KV Cache 内存计算详解

## 代码位置

`nanovllm/engine/model_runner.py:100-112`

```python
def allocate_kv_cache(self):
    config = self.config
    hf_config = config.hf_config

    # 获取 GPU 内存信息
    free, total = torch.cuda.mem_get_info()
    used = total - free                                                      # A
    peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]            # B
    current = torch.cuda.memory_stats()["allocated_bytes.all.current"]      # C

    # 计算 KV cache 参数
    num_kv_heads = hf_config.num_key_value_heads // self.world_size
    head_dim = getattr(hf_config, "head_dim",
                       hf_config.hidden_size // hf_config.num_attention_heads)

    # 单个 block 的字节数
    block_bytes = (2 * hf_config.num_hidden_layers * self.block_size *
                   num_kv_heads * head_dim * hf_config.torch_dtype.itemsize)

    # 可分配的 block 数量
    config.num_kvcache_blocks = int(total * config.gpu_memory_utilization
                                     - used - peak + current) // block_bytes

    # 预分配 KV cache
    self.kv_cache = torch.empty(2, hf_config.num_hidden_layers,
                                 config.num_kvcache_blocks, self.block_size,
                                 num_kv_heads, head_dim)
```

## 逐步解析

### 第 1 步：获取 GPU 内存状态

```python
free, total = torch.cuda.mem_get_info()
used = total - free
peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
```

| 变量 | 含义 | 示例值 (24GB GPU) |
|------|------|------------------|
| `total` | GPU 总显存 | 24 GB |
| `free` | 当前可用显存 | 22 GB |
| `used` | 已占用显存 (= total - free) | 2 GB |
| `peak` | 历史峰值分配（模型权重 + warmup 激活值） | 6 GB |
| `current` | 当前已分配（主要是模型权重） | 2 GB |

**时序说明**：

```
执行流程：
┌─────────────────────────────────────────────────────────────┐
│ 1. __init__ (第31行)                                         │
│    └─ 加载模型权重 → current ≈ 2 GB                          │
├─────────────────────────────────────────────────────────────┤
│ 2. warmup_model() (第34行)                                   │
│    └─ 运行最大长度序列                                       │
│       ├─ 中间激活值占用 ~4 GB                                │
│       └─ peak = 2 + 4 = 6 GB                                │
│       └─ warmup 结束后激活值释放                            │
│       └─ current 恢复到 ~2 GB                               │
├─────────────────────────────────────────────────────────────┤
│ 3. allocate_kv_cache() (第35行) ← 当前位置                   │
│    └─ 此时 current ≈ 2 GB (仅模型权重)                       │
│    └─ peak = 6 GB (记录了峰值)                              │
└─────────────────────────────────────────────────────────────┘
```

### 第 2 步：计算每个 GPU 的 KV head 数

```python
num_kv_heads = hf_config.num_key_value_heads // self.world_size
```

**张量并行**：将 KV heads 平均分配到各个 GPU。

| 配置 | 总头数 | GPU 数量 | 每个 GPU |
|------|--------|----------|----------|
| `num_key_value_heads` | 24 | 1 | 24 |
| `num_key_value_heads` | 24 | 2 | 12 |
| `num_key_value_heads` | 24 | 4 | 6 |

### 第 3 步：获取 head 维度

```python
head_dim = getattr(hf_config, "head_dim",
                   hf_config.hidden_size // hf_config.num_attention_heads)
```

**公式**：

```
head_dim = hidden_size / num_attention_heads
```

**示例** (Qwen3-0.6B):

| 参数 | 值 |
|------|-----|
| `hidden_size` | 1536 |
| `num_attention_heads` | 24 |
| `head_dim` | 64 |

### 第 4 步：计算单个 block 的字节数

```python
block_bytes = (2 * hf_config.num_hidden_layers * self.block_size *
               num_kv_heads * head_dim * hf_config.torch_dtype.itemsize)
```

**公式展开**：

```
block_bytes = 2                      # K 和 V 各一份
             × num_layers            # 每层都需要 KV cache
             × block_size            # 每个 block 的 token 数（默认 256）
             × num_kv_heads          # 每个 GPU 的 KV head 数
             × head_dim              # 每个 head 的维度
             × dtype.itemsize        # 每个元素的字节数（fp16=2, fp32=4）
```

**数值示例** (Qwen3-0.6B, 单 GPU, FP16):

| 参数 | 值 |
|------|-----|
| `num_layers` | 24 |
| `block_size` | 256 |
| `num_kv_heads` | 24 |
| `head_dim` | 64 |
| `dtype.itemsize` | 2 (FP16) |

```
block_bytes = 2 × 24 × 256 × 24 × 64 × 2
            = 47,185,920 字节
            ≈ 45 MB
```

### 第 5 步：计算可分配的 block 数量

```python
available_bytes = int(total * config.gpu_memory_utilization - used - peak + current)
num_kvcache_blocks = available_bytes // block_bytes
```

**公式推导**：

```
available = total × gpu_memory_utilization - used - peak + current
          = 目标显存 - used - (peak - current) - current + current
          = 目标显存 - used - 峰值激活值
```

**关键理解**：
- `peak - current` ≈ warmup 时的**峰值激活值大小** (临时内存，已释放)
- `used` ≈ `current` + 缓存碎片等未归还 CUDA 的内存
- 计算目的是为 KV cache 预留足够空间，同时考虑峰值激活值

**数值示例** (24GB GPU, 90% 利用率):

| 变量 | 值 | 说明 |
|------|-----|------|
| `total` | 24 GB | 总显存 |
| `gpu_memory_utilization` | 0.9 | 目标使用率 |
| `used` | 2.5 GB | 当前占用（含碎片） |
| `peak` | 6 GB | 历史峰值 |
| `current` | 2 GB | 当前分配 |
| `peak - current` | 4 GB | 峰值激活值 |

```
available = 24 × 0.9 - 2.5 - 6 + 2
          = 21.6 - 2.5 - 6 + 2
          = 15.1 GB

num_blocks = 15.1 GB / 45 MB ≈ 337 blocks

总 KV cache 容量 = 337 × 256 tokens ≈ 86,272 tokens
```

## 内存布局示意图

```
GPU 显存分配 (24 GB):
┌────────────────────────────────────────────┐
│  模型权重                  │ 2-4 GB   │
├────────────────────────────────────────────┤
│  中间激活值 (峰值)         │ 3-5 GB   │ ← warmup 时峰值
├────────────────────────────────────────────┤
│  KV Cache (可变长度)          │ 10-16 GB│ ← 动态计算
│  ┌─────┬─────┬─────┬─────┬─────┬───┐    │
│  │Block│Block│Block│Block│Block│...│    │
│  │ 256 │ 256 │ 256 │ 256 │ 256 │   │    │
│  │tokens│tokens│tokens│tokens│tokens│   │    │
│  └─────┴─────┴─────┴─────┴─────┴───┘    │
├────────────────────────────────────────────┤
│  预留空间 (10%)                │ ~2.4 GB  │
└────────────────────────────────────────────┘
```

## KV Cache 的张量形状

```python
self.kv_cache = torch.empty(
    2,                      # K 和 V
    num_layers,             # 每层独立
    num_blocks,             # 物理块数量
    block_size,             # 每块的 token 数
    num_kv_heads,           # 每个 GPU 的 KV head 数
    head_dim                # 每个 head 的维度
)
```

**示例** (Qwen3-0.6B, 单 GPU, 337 blocks, FP16):

```python
kv_cache.shape = [2, 24, 337, 256, 24, 64]
                 ↑  ↑  ↑    ↑    ↑   ↑
                 │  │  │    │    │   └─ head_dim = 64
                 │  │  │    │    └────── num_kv_heads = 24
                 │  │  │    └─────────── block_size = 256
                 │  │  └──────────────── num_blocks = 337
                 │  └──────────────────── num_layers = 24
                 └──────────────────────── K/V (2 份)
```

---

# 多头注意力架构对比

## 基本概念

在注意力机制中：
- **Attention heads (Q heads)**: Query 的投影头数
- **KV heads**: Key 和 Value 的投影头数

## 三种注意力架构的演进

### 论文来源与历史

注意力机制的发展经历了三个重要阶段：

| 架构 | 论文 | 作者 | 发表时间 | 引用量* |
|------|------|------|----------|---------|
| **MHA** | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Vaswani et al. (Google) | 2017.06 | 200,000+ |
| **MQA** | (多项工作的实践总结) | - | 2019-2021 | - |
| **GQA** | [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) | Ainslie et al. | 2023.05 | 1,385+ |

\* 引用量截至 2025 年底的统计数据

### 架构对比矩阵

```
┌─────────────────────────────────────────────────────────────────┐
│                        注意力架构演进                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  MHA (2017)     MQA (2019-21)     GQA (2023)                    │
│  ┌─────────┐    ┌─────────┐       ┌─────────┐                   │
│  │ Q1 → K1 │    │ Q1 ┐    │       │ Q1 Q2 ┐ │                   │
│  │ Q2 → K2 │    │ Q2 ┤    │       │ Q3 Q4 ├→ K1                 │
│  │ Q3 → K3 │    │ Q3 ┤    │       │ Q5 Q6 ┘ │                   │
│  │ ...     │    │ ... ├→ K │       │         │                   │
│  │ Qn → Kn │    │ Qn ┘    │       │  ...   → K2                 │
│  └─────────┘    └─────────┘       └─────────┘                   │
│  表达能力强      极致性能           平衡                        │
│  内存占用大      质量下降           质量接近 MHA                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 三种注意力架构详解

### 1. MHA (Multi-Head Attention) - 传统多头注意力

**论文基础**：[Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)

**核心思想**：

Multi-Head Attention 允许模型**同时关注不同位置的表示子空间**。论文指出：

> "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions."

**数学定义**：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

其中：

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

**关键创新点**：
- **多个独立的 attention heads**：每个 head 学习不同的表示子空间
- **完整的 Q、K、V 投影**：每个 head 有独立的权重矩阵 $W_i^Q, W_i^K, W_i^V$
- **输出投影**：concatenation 后通过 $W^O$ 融合

**配置**：

```python
num_attention_heads = 24
num_key_value_heads = 24  # Q 和 KV 的头数相同
```

**结构**：

```
每个 Q head 都有自己独立的 K、V head:

Q1 → K1, V1
Q2 → K2, V2
Q3 → K3, V3
...
Q24 → K24, V24
```

**KV cache 大小**：

```
KV_cache = num_layers × seq_len × num_kv_heads × head_dim × 2 bytes
```

**优点**：
- ✅ 表达能力最强
- ✅ 每个 head 可以学习不同的注意力模式

**缺点**：
- ❌ KV cache 内存占用大
- ❌ Decode 阶段内存带宽压力大

**论文原文引用**：

> "Instead of performing a single attention function, the authors found it beneficial to linearly project the queries, keys and values h times with different learned linear projections... The outputs are concatenated and once again projected."

**实践影响**：MHA 成为 Transformer 模型的标准配置，被 BERT、GPT、T5 等主流模型广泛采用。

---

### 2. MQA (Multi-Query Attention) - 单 KV 头

**背景**：MQA 不是由单一论文提出的，而是 2019-2021 年期间多项工作的**实践总结**：
- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) (Shazeer, 2019)
- [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311) (Chowdhery et al., 2022)
- [Transformer-XL](https://arxiv.org/abs/1901.02860) (Dai et al., 2019) - 类似思想

**核心思想**：

**所有 query heads 共享同一个 key-value head**，大幅减少 KV cache 内存占用。

GQA 论文对 MQA 的总结：

> "Multi-query attention (MQA), which only uses a single key-value head, drastically speeds up decoder inference."

**设计动机**：

1. **推理瓶颈**：Decode 阶段，每生成一个 token 都需要读取完整的 KV cache
2. **内存带宽限制**：KV cache 读取成为主要瓶颈，而非计算
3. **观察**：多个 KV heads 的信息可以通过单个 KV head 近似

**配置**：

```python
num_attention_heads = 24
num_key_value_heads = 1   # 所有 Q head 共享一对 K、V
```

**结构**：

```
所有 Q heads 共享同一个 K、V:

Q1 ┐
Q2 ┤
Q3 ┤
... ├→ K1, V1 (共享)
Q24┘
```

**KV cache 大小**：

```
KV_cache = num_layers × seq_len × 1 × head_dim × 2 bytes
```

**内存节省**：24 倍 (相比 24 个 KV heads)

**优点**：
- ✅ KV cache 极小
- ✅ Decode 阶段吞吐量高

**缺点**：
- ❌ 表达能力下降
- ❌ 可能影响生成质量

**GQA 论文指出的问题**：

> "However, MQA can lead to quality degradation, and moreover it may not be desirable to train a separate model just for faster inference."

**实际应用**：
- **PaLM 540B**: Google 的超大规模语言模型采用 MQA
- **BLOOM**: BigScience 开源模型使用类似设计
- **推理加速**：工业界广泛用于提升吞吐量

---

### 3. GQA (Grouped Query Attention) - 分组查询注意力

**论文基础**：[GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) (Ainslie et al., 2023, EMNLP)

**核心思想**：

GQA 是 **MHA 和 MQA 的插值**，将 query heads 分组，每组共享一个 KV head：

$$\text{num\_groups} = G, \quad 1 < G < \text{num\_heads}$$

**论文摘要**：

> "We introduce grouped-query attention (GQA), a generalization of multi-query attention which uses an intermediate (more than one, less than number of query heads) number of key-value heads. We show that uptrained GQA achieves quality close to multi-head attention with comparable speed to MQA."

**关键创新**：

1. **Uptraining 技术**：从预训练的 MHA 模型转换到 GQA，只需 **5% 的原始预训练算力**

   > "We propose a recipe for uptraining existing multi-head language model checkpoints into models with MQA using 5% of original pre-training compute."

2. **分组插值**：通过调整分组数量在质量和性能间取得平衡

3. **无需从零训练**：可以直接转换现有的 MHA 模型

**数学形式**：

设有 $h$ 个 query heads，分成 $g$ 组，每组有 $k = h/g$ 个 queries 共享一对 KV：

$$\text{Group}(i) = \lfloor i / k \rfloor$$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_{\text{Group}(i)}^K, VW_{\text{Group}(i)}^V)$$

**配置**：

```python
num_attention_heads = 24
num_key_value_heads = 4   # 每 6 个 Q heads 共享一对 K、V
```

**结构**：

```
Q heads 分组共享 K、V:

Q1, Q2, Q3, Q4, Q5, Q6    ─→ K1, V1  (组 1)
Q7, Q8, Q9, Q10, Q11, Q12 ─→ K2, V2  (组 2)
Q13, Q14, Q15, Q16, Q17, Q18 ─→ K3, V3  (组 3)
Q19, Q20, Q21, Q22, Q23, Q24 ─→ K4, V4  (组 4)
```

**KV cache 大小**：

```
KV_cache = num_layers × seq_len × 4 × head_dim × 2 bytes
```

**内存节省**：6 倍 (相比 24 个 KV heads)

**优点**：
- ✅ 内存占用和性能的平衡
- ✅ 生成质量接近 MHA

**缺点**：
- ⚠️ 需要调优分组数量

## 在 nano-vLLM 中的体现

### 张量并行后的 KV head 数量

```python
# model_runner.py:106
num_kv_heads = hf_config.num_key_value_heads // self.world_size
```

**示例 1** (Qwen3-0.6B, 单卡):

| 配置 | 总头数 | 每个 GPU | KV head 组 |
|------|--------|----------|------------|
| `num_attention_heads` | 24 | 24 (Q) | - |
| `num_key_value_heads` | 24 | 24 (KV) | 1:1 (MHA) |

**示例 2** (Qwen3-0.6B, 2卡):

| 配置 | 总头数 | 每个 GPU | KV head 组 |
|------|--------|----------|------------|
| `num_attention_heads` | 24 | 12 (Q) | - |
| `num_key_value_heads` | 24 | 12 (KV) | 1:1 (MHA) |

**示例 3** (Qwen-14B, 2卡, GQA):

| 配置 | 总头数 | 每个 GPU | KV head 组 |
|------|--------|----------|------------|
| `num_attention_heads` | 40 | 20 (Q) | - |
| `num_key_value_heads` | 8 | 4 (KV) | 5:1 (GQA) |

### 对内存的影响

在 `allocate_kv_cache()` 中：

```python
block_bytes = 2 * num_layers * block_size * num_kv_heads * head_dim * dtype_size
```

**对比** (同一模型，24GB GPU, FP16):

| 架构 | `num_kv_heads` | `block_bytes` | `num_kvcache_blocks` | 总容量 | 内存节省 |
|------|----------------|---------------|----------------------|--------|----------|
| **MHA** | 12 | 45 MB | 337 | 86K tokens | - |
| **GQA** | 4 | 15 MB | 1013 | 259K tokens | **3x** |
| **MQA** | 1 | 3.75 MB | 4049 | 1.04M tokens | **12x** |

## 为什么区分 Q 和 KV heads？

### 内存优势

```
KV cache 大小 ∝ num_key_value_heads

对于 seq_len = 4096, num_layers = 24, head_dim = 64, dtype = fp16:

MHA (24 KV heads):
  24 × 24 × 4096 × 64 × 2 = 3.01 GB

GQA (4 KV heads):
  4 × 24 × 4096 × 64 × 2 = 0.50 GB  ← 6x 内存节省

MQA (1 KV head):
  1 × 24 × 4096 × 64 × 2 = 0.13 GB  ← 23x 内存节省
```

### 计算优势

在 Decode 阶段，每生成一个 token 需要：

```python
# 伪代码
for each layer:
    # 1. 计算 Q (1 token)
    q = compute_q(new_token)  # shape: [1, num_q_heads, head_dim]

    # 2. 从 KV cache 读取 (seq_len tokens)
    k = k_cache[:, :seq_len, :, :]  # shape: [seq_len, num_kv_heads, head_dim]
    v = v_cache[:, :seq_len, :, :]

    # 3. 计算 attention
    attn = flash_attn_with_kvcache(q, k, v, ...)
```

**瓶颈分析**：

- **步骤 1**：计算量小 (1 token)
- **步骤 2**：**内存带宽密集**，需要读取大量 KV cache
  - `seq_len × num_kv_heads × head_dim × 2 bytes`
- **步骤 3**：计算量中等 (O(seq_len))

**关键发现**：

```
内存读取量 ∝ num_kv_heads

较少的 KV heads =:
  ✅ 更少的内存带宽需求
  ✅ 更高的 Decode 吞吐量
  ✅ 支持更长上下文
```

### 实际性能对比

**理论吞吐量** (Decode 阶段，假设内存带宽 2 TB/s):

| 架构 | `num_kv_heads` | 每次 token 读取量 | 受内存带宽限制的吞吐量 |
|------|----------------|------------------|---------------------|
| MHA | 24 | 48 MB | ~43K tokens/s |
| GQA | 4 | 8 MB | ~250K tokens/s |
| MQA | 1 | 2 MB | ~1M tokens/s |

**注意**：实际吞吐量还受其他因素影响（kernel 性能、batch size 等）。

## 主流模型的架构选择

| 模型 | 架构 | `num_q_heads` | `num_kv_heads` | 分组比 |
|------|------|---------------|----------------|--------|
| **LLaMA 2/3** | GQA | 32 | 8 | 4:1 |
| **Mistral 7B** | GQA | 32 | 8 | 4:1 |
| **Qwen2/3** | GQA | 28 | 4 (7B) / 8 (14B+) | 7:1 / 3.5:1 |
| **Falcon** | MQA | 71 | 1 | 71:1 |
| **GPT-3** | MHA | 96 | 96 | 1:1 |

**趋势**：现代模型普遍采用 GQA，在质量和性能间取得平衡。

---

# 总结

## KV Cache 内存计算要点

1. **时序**：先 warmup 测峰值，再分配 KV cache
2. **公式**：
   ```
   available = total × utilization - used - (peak - current)
   num_blocks = available / block_bytes
   ```
3. **block 大小**：
   ```
   block_bytes = 2 × layers × block_size × kv_heads × head_dim × dtype
   ```

## 多头注意力架构选择

| 架构 | 适用场景 |
|------|----------|
| **MHA** | 追求最佳质量，内存充足 |
| **GQA** | **主流选择**，平衡质量和性能 |
| **MQA** | 极致吞吐需求，可接受质量损失 |

## 实践建议

1. **模型选择**：优先使用 GQA 架构的模型
2. **内存规划**：根据 `num_kv_heads` 预估 KV cache 需求
3. **性能调优**：MQA/GQA 可支持更大 batch size

---

# 参考资源

## 核心论文

### Multi-Head Attention (MHA)
- **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)**
  - Vaswani et al. (Google), 2017
  - NeurIPS 2017
  - 引用量: 200,000+
  - **原始论文**: [PDF](https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf)

### Multi-Query Attention (MQA)
- **[Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)**
  - Noam Shazeer, 2019
  - 引用量: 1,200+
- **[PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311)**
  - Chowdhery et al. (Google), 2022
  - 引用量: 4,500+
- **[Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860)**
  - Dai et al., 2019
  - 引用量: 3,800+

### Grouped-Query Attention (GQA)
- **[GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)**
  - Ainslie et al., 2023
  - EMNLP 2023
  - 引用量: 1,385+
  - **原始论文**: [arXiv](https://arxiv.org/abs/2305.13245) | [PDF](https://arxiv.org/pdf/2305.13245)

## 相关技术

### Flash Attention
- **[Flash Attention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)**
  - Dao et al., 2022
  - 引用量: 2,500+

### PagedAttention
- **[Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)**
  - Kwon et al. (vLLM team), 2023
  - 引用量: 1,200+

## 项目资源

- **[Nano-vLLM GitHub](https://github.com/GeeeekExplorer/nano-vllm)**: 本文档对应的代码库
- **[vLLM GitHub](https://github.com/vllm-project/vllm)**: 生产级 LLM 推理引擎
- **[Flash Attention Implementation](https://github.com/Dao-AILab/flash-attention)**: 官方实现

## 学习资源

### MHA 详解
- [Multi-Head Attention Mechanism - GeeksforGeeks](https://www.geeksforgeeks.org/nlp/multi-head-attention-mechanism/)
- [Attention Is All You Need: The Paper That Revolutionized AI - Medium](https://medium.com/@vijay.poudel1/attention-is-all-you-need-the-paper-that-revolutionized-ai-6e606e6a847b)
- [【重温经典】Attention is all you need - 知乎](https://zhuanlan.zhihu.com/p/638884759)

### MQA/GQA 详解
- [Multi-Query Attention Explained - TowardsAI](https://pub.towardsai.net/multi-query-attention-explained-844dfc4935bf)
- [Understanding Attention and Multi Query Attention - Medium](https://medium.com/@qinliu.cn/understanding-attention-and-multi-query-attention-7b931fd10e53)
- [Multi-Query Attention is All You Need - Fireworks AI](https://fireworks.ai/blog/multi-query-attention-is-all-you-need)
- [Grouped-Query Attention(GQA) Explained - Substack](https://aiexpjourney.substack.com/p/grouped-query-attention-gqa-explained-5f3dbbfe013b)
- [A Gentle Introduction to Multi-Head Attention and Grouped-Query Attention - Machine Learning Mastery](https://machinelearningmastery.com/a-gentle-introduction-to-multi-head-attention-and-grouped-query-attention/)

### KV Cache 和内存管理
- [vLLM 官方文档 - CUDA Graphs](https://docs.vllm.ai/en/stable/design/cuda_graphs/)
- [NVIDIA CUDA Graphs Performance Blog (2024)](https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements/)
- [NVIDIA TensorRT Best Practices](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html)

## 硬件规格
- [NVIDIA H100 GPU 规格](https://www.nvidia.com/en-us/data-center/h100/)
- [NVIDIA A100 GPU 规格](https://www.nvidia.com/en-us/data-center/a100/)
- [H200 vs H100 vs A100 性能对比](https://xconnectglobal.com/2025/11/06/nvidia-a100-vs-h100-vs-h200/)
