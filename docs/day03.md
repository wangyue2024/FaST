# Day 3 攻坚计划：Warp Level Primitives 线程束原语与并行归约 (Parallel Reduction)

---

## 1. Day 3 核心定位与时间块规划 (Daily Schedule)

Day 3 的核心目标是 **“深入理解 SM 内部 Warp 级硬件原语（Warp Primitives） + 攻克线程分化 (Warp Divergence) 与 Shared Memory 屏障开销 + 用 Warp Shuffle 指令、指令重排 (ILP)、Block 连续 Tile 切片与 float4 向量化实现极致性能的并行归约（Parallel Reduction）”**。

| 时间段 | 模块 | 核心任务 | 交付标准 |
| :--- | :--- | :--- | :--- |
| **Block 1 (2.0h)** | **理论与微架构** | 精读 Warp 级指令与硬件掩码；推导并行树状归约演进脉络与 Bank Conflict / RAW 依赖 / DRAM Row Buffer / 向量化 128-bit 指令。 | 输出包含 Warp Shuffle 原理与 10 代 Reduction 演进逻辑的结构化笔记。 |
| **Block 2 (3.0h)** | **CUDA 核心编码实战** | 建立 `src/03_reduction/main.cu`，实现 10 个版本的 Reduction Kernel（从 Baseline 到 float4 128-bit 向量化）。 | 代码通过编译，通过 CPU 归约精度校验，输出 GFLOPS 与 GB/s 对比。 |
| **Block 3 (1.0h)** | **性能分析与复盘** | 测量 10 种版本的有效显存带宽（GB/s），分析 Divergence、Shuffle 指令、ILP、Tile 连续访存与 float4 向量化带来的加速比。 | 完成 `docs/day03.md` 复盘总结并提交 Git Commit。 |

---

## 2. 核心知识点拆解 (Core Knowledge Points)

### 2.1 知识点 1：Warp-Level Primitives (线程束级原语与寄存器交换)

#### 1. 为什么需要 Warp Shuffle？
* 在传统的 Shared Memory 方案中，线程 A 想传数据给线程 B，必须：
  `线程 A 写 Shared Mem` $\rightarrow$ `__syncthreads()` 屏障等待 $\rightarrow$ `线程 B 读 Shared Mem`。
* **Warp Shuffle 指令**：允许同一个 Warp 内部的 32 个线程**直接在片上寄存器（Register File）之间交换数据**！
  * **延迟**：仅需 **1 个时钟周期**！
  * **零 Shared Memory 占用**，**零 Block 屏障同步开销**！

---

### 2.2 知识点 2：Vectorized Memory Access (128-bit float4 向量化访存)

* **原理**：将 `float*` 强转为 `const float4*`，一条指令直接发射 `LDG.E.128` (128-bit / 16 字节) 向量读取。
* **优势**：指令发射开销直接减少 75%，大幅提升 SM 译码吞吐量与显存总线满载效率！

---

### 2.3 知识点 3：Parallel Reduction 10 代演进图谱

```plaintext
[v1. 朴素交错] ──► [v2. 连续线程] ──► [v3. 反向步长] ──► [v4. First Add] ──► [v5. Unroll Last Warp]
                                                                                      │
[v10. float4 向量化] ◄──(128-bit LDG)─── [v9. Block Tile] ◄─── [v8. ILP 重排] ◄─── [v7. 终极融合] ◄─── [v6. Warp Shuffle]
```

---

## 3. Day 3 任务要求与编码落地方案

### 3.1 编码任务：`src/03_reduction/main.cu`

实现并对比 **10 个版本的 Reduction 算子**：

1. **`kernel_reduce_v1_divergent`**：朴素交错归约（存在 Divergence）。
2. **`kernel_reduce_v2_sequential`**：连续线程归约（存在 Shared Mem Bank Conflict）。
3. **`kernel_reduce_v3_interleaved`**：反向步长归约（消除 Bank Conflict）。
4. **`kernel_reduce_v4_first_add`**：Load 阶段首加（2x 数据搬运压缩）。
5. **`kernel_reduce_v5_unroll_last`**：展开 Last Warp 循环。
6. **`kernel_reduce_v6_shuffle`**：基于 `__shfl_down_sync` 寄存器极速归约（受限于 RAW 依赖）。
7. **`kernel_reduce_v7_ultimate`**：1KB Coalesced 首加 Load + Warp Shuffle 寄存器归约 (终极融合版)。
8. **`kernel_reduce_v8_ilp_shuffle`**：多累加器 (ILP=4) 指令重排 + Warp Shuffle (消除 RAW 依赖，全速读取)。
9. **`kernel_reduce_v9_block_contiguous`**：Block 连续 Tile 切片读取 + ILP + Shuffle 归约 (100% DRAM Row Buffer 命中)。
10. **`kernel_reduce_v10_vectorized_float4`**：128-bit `float4` 向量化读取 + Warp Shuffle 归约 (减少 75% 指令开销)。

### 3.2 实测基准性能表格 (RTX 5060, 1000 次循环, N=16M)

| 算子变体 | 耗时 (ms) | 显存带宽 (GB/s) | 加速比 | 核心机制 / 瓶颈 |
| :--- | :--- | :--- | :--- | :--- |
| **1. Reduce v1 (Naive Divergent)** | 0.658 | 102.0 | 1.00x | 取模导致严重 Warp Divergence |
| **2. Reduce v2 (Sequential Threads)** | 0.417 | 160.8 | 1.58x | 消除 Divergence，存在 Bank Conflict |
| **3. Reduce v3 (Interleaved Addressing)** | 0.401 | 167.4 | 1.64x | 消除 Bank Conflict |
| **4. Reduce v4 (First Add on Load)** | 0.213 | 314.8 | 3.09x | Global Load 时先做首加，省一半 Shared Mem |
| **5. Reduce v5 (Unroll Last Warp)** | 0.172 | 389.7 | 3.82x | 展开末尾 Warp，海量 Block 并发打满 DRAM 带宽 |
| **6. Reduce v6 (Warp Shuffle Extreme)** | 0.183 | 366.3 | 3.59x | 寄存器归约，但单累加器存在 RAW 依赖 |
| **7. Reduce v7 (Ultimate Hybrid)** | **0.170** 🚀 | **395.2** 🔝 | **3.87x** | **连续 1KB Coalesced Load + Warp Shuffle (终极形态)** |
| **8. Reduce v8 (ILP Reorder Shuffle)** | **0.172** ⚡ | **390.9** 🔝 | **3.83x** | **多累加器 (ILP=4) 指令重排，消灭 RAW 依赖** |
| **9. Reduce v9 (Block-Contiguous Tile)** | **0.172** ⚡ | **390.8** 🔝 | **3.83x** | **Block 连续 Tile 切片，100% DRAM Row Buffer 命中** |
| **10. Reduce v10 (Vectorized float4)** | **0.172** ⚡ | **391.2** 🔝 | **3.83x** | **128-bit float4 向量化读取，指令开销降低 75%** |

---

## 4. 交付检查清单 (Checklist)

| 检查项 | 交付标准 |
| :--- | :--- |
| **理论准备** | 理解 Warp Shuffle 四大原语与树状归约消除 Divergence / Bank Conflict / RAW 依赖 / DRAM Row 切换 / float4 向量化原理 |
| **代码实现** | 成功在 `src/03_reduction/main.cu` 中实现 v1 ~ v10 完整 10 代 Reduction 算子 |
| **精度验证** | 10 种算子均通过 CPU 参考归约的 `[ PASS ]` 校验 |
| **性能提升** | float4 向量化 (v10)、Tile 切片版 (v9)、ILP 重排版 (v8) 与 Ultimate (v7) 显存带宽均达到 390+ GB/s (加速比 3.83x+) |
| **复盘文档** | 在 `docs/day03.md` 中记录 10 代算子的 1000 次 Stable 耗时与 GB/s 带宽对比 |
| **Git 提交** | 完成规范 Commit 记录：`feat(day03): implement parallel reduction v1~v10 benchmark` |
