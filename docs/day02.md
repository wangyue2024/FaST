# Day 2 攻坚计划：Shared Memory 共享内存与 32-Bank Conflict 破解

---

## 1. Day 2 核心定位与时间块规划 (Daily Schedule)

Day 2 的核心目标是 **“理解 SM 片上 SRAM 共享内存微架构 + 攻克 32-Bank Conflict 冲突 + 用 Padding 技巧实现极致性能的矩阵转置”**。

通过 Shared Memory，我们将解决 Day 1 中矩阵转置存在的“非合并写（Non-Coalesced Write）”瓶颈，实现**“连续读 + 连续写”**的双重合并访存！

| 时间段 | 模块 | 核心任务 | 交付标准 |
| :--- | :--- | :--- | :--- |
| **Block 1 (2.0h)** | **理论与硬件微架构** | 精读 CUDA Guide 中 Shared Memory 与 Bank 结构；白板推导 32-Bank 映射公式与 Bank Conflict 物理成因。 | 输出包含 Bank Conflict 冲突与 Padding 错位避让原理的结构化笔记。 |
| **Block 2 (3.0h)** | **CUDA 核心编码实战** | 编写 `src/02_transpose/main.cu`，实现 3 个版本的转置 Kernel（Naive, Shared Memory, Padded Shared Memory）。 | 代码通过编译，通过 CPU 转置精度校验，输出带宽对比。 |
| **Block 3 (1.0h)** | **性能分析与复盘** | 测量 3 种版本的有效显存带宽（GB/s），分析 Shared Memory 带来的加速比与 Bank 冲突消除效果。 | 完成 `docs/day02.md` 复盘总结并提交 Git Commit。 |

---

## 2. 核心知识点拆解 (Core Knowledge Points)

### 2.1 知识点 1：Shared Memory 物理微架构
* **物理位置：** 位于 SM 内部的片上高速存储器（On-chip SRAM），与 L1 Data Cache 共享物理 SRAM 资源（通常为 64KB ~ 128KB）。
* **访问延迟：** 仅需 **20 ~ 30 个时钟周期**（比外部 DRAM 物理显存快 **$20 \times$** 以上）。
* **生命周期与作用域：** 声明为 `__shared__` 的变量生命周期与线程块（Block）一致，由同一个 Block 内部的所有线程共同共享与读写。
* **核心作用：** 
  1. **数据复用 (Data Reuse)：** 避免重复去慢速 Global Memory 读取数据。
  2. **访存重构 (Memory Access Reordering)：** 将 Global Memory 的非合并访存转换为片上 Shared Memory 的中转，从而实现 Global Memory 的纯合并读写。

---

### 2.2 知识点 2：32-Bank 物理存储结构与 Bank Conflict 成因

#### 1. 32-Bank 映射公式
为了在同一个时钟周期内支持 32 个线程（1 个 Warp）同时并发读写，GPU 硬件将 Shared Memory 在物理上拆分成了 **32 个独立演播的存储块（Banks）**（Bank 0 到 Bank 31）。

* 每个 Bank 的宽度为 **4 Bytes (32 bits)**（刚好对应 1 个 `float` 或 `int`）。
* **地址映射公式：**
  $$\text{Bank ID} = \left( \frac{\text{Byte Address}}{4} \right) \pmod{32} = \text{Element Index} \pmod{32}$$

#### 2. 三种 Bank 访问状态
当 1 个 Warp 32 个线程同时发起 Shared Memory 读写时：
* **无冲突 (No Conflict - 最快)：** 32 个线程分别访问 32 个不同的 Banks $\rightarrow$ **1 个周期内全并发完成！**
* **广播 (Broadcast - 极快)：** 多个线程同时读取同一个 Bank 的**同一个 4-Byte 地址** $\rightarrow$ 硬件触发广播机制，**1 个周期内完成！**
* **Bank 冲突 (Bank Conflict - 严重惩罚)：** 多个线程访问同一个 Bank 的**不同 4-Byte 地址** $\rightarrow$ 硬件请求无法并发，必须**被迫串行化（Serialization）**。若是 32-way Bank Conflict，读取延迟将直接**暴增 32 倍**！

---

### 2.3 知识点 3：利用 Shared Memory 解决矩阵转置非合并写

Day 1 的 Naive 矩阵转置中：
* 读取连续 $A[y \times W + x]$（合并读），但写入跨步 $C[x \times H + y]$（非合并写）。

**Day 2 Shared Memory 优化方案（两步中转法）：**

```plaintext
Global Memory (连续读 A) ──► Shared Memory Tile[y][x] ──(转置坐标)──► Global Memory (连续写 C)
```

1. **合并读取：** 32 个线程从 Global Memory **连续读取**数据，填入 `__shared__ float tile[32][32]`。
2. **块内同步：** 调用 `__syncthreads()` 确保整个 Block 内的数据已完全载入片上。
3. **合并写入：** 颠倒坐标，从 Shared Memory 读取 `tile[threadIdx.x][threadIdx.y]`，**连续写入** Global Memory！

---

### 2.4 知识点 4：Padding 避让技巧（彻底消除 32-way Bank Conflict）

#### 冲突产生原因：
如果在 Shared Memory 中声明正方形二维数组 `__shared__ float tile[32][32]`：
* 写入 `tile[threadIdx.y][threadIdx.x]` 时，按行写入，无 Bank Conflict。
* 当从 Shared Memory 按列读取 `tile[threadIdx.x][threadIdx.y]` 时：
  * 线程 0 读取 `tile[0][0]`（地址索引 0 $\rightarrow$ Bank 0）
  * 线程 1 读取 `tile[1][0]`（地址索引 $1 \times 32 = 32 \rightarrow$ Bank 0）
  * 线程 2 读取 `tile[2][0]`（地址索引 $2 \times 32 = 64 \rightarrow$ Bank 0）
* **整条 Warp 的 32 个线程全部命中 Bank 0！引发严重的 32-way Bank Conflict 串行化惩罚！**

#### Padding 避让解法（简单而伟大）：
只需要在声明 Shared Memory 时，在列维度**额外增加 1 列空用填充（Padding）**：

```cpp
// ❌ 产生 32-way Bank Conflict:
__shared__ float tile[32][32];

// ✅ 彻底消除 Bank Conflict (Padding 技巧):
__shared__ float tile[32][33]; 
```

**数学推导原理：**
跨度由 32 变成了 33：
* 线程 0 访问 `tile[0][0]` $\rightarrow$ 索引 $0 \rightarrow \mathbf{Bank \ 0}$
* 线程 1 访问 `tile[1][0]` $\rightarrow$ 索引 $1 \times 33 = 33 \rightarrow \mathbf{Bank \ 1}$
* 线程 2 访问 `tile[2][0]` $\rightarrow$ 索引 $2 \times 33 = 66 \rightarrow \mathbf{Bank \ 2}$

**错开 1 列后，32 个线程瞬间精准错落映射到了 32 个不同的 Banks 上，Bank Conflict 被彻底降到了 0！**

---

## 3. Day 2 任务要求与代码落地方案

### 3.1 编码任务：`src/02_transpose/main.cu`

建立 `src/02_transpose/main.cu`，实现并对比 **3 个版本的转置算子**：

1. **`kernel_transpose_naive`：** Day 1 朴素转置（读连续，写跨步）。
2. **`kernel_transpose_shared`：** 基于 `__shared__ float tile[32][32]` 中转（无 Padding，存在 32-way Bank Conflict）。
3. **`kernel_transpose_padded`：** 基于 `__shared__ float tile[32][33]` 中转（带 Padding，完全无 Bank Conflict）。

### 3.2 代码与测试流程规范

* **工程包含：** 引入 [include/cuda_timer.h](file:///mnt/d/1file/Desktop/code/FaST/include/cuda_timer.h) 进行高精度 `CudaTimer` 计时。
* **精度校验：** CPU 参考实现 `transposeCPU`，断言 $|gpu[i] - cpu[i]| < 1e-5$。
* **尺寸规模：** 测试 $4096 \times 4096$ 矩阵（单矩阵 64 MB，击穿 L2 Cache 展现 DRAM 物理真实性能）。

---

## 4. 交付检查清单 (Checklist)

| 检查项 | 交付标准 |
| :--- | :--- |
| **代码实现** | 成功在 `src/02_transpose/main.cu` 中实现 3 种版本的转置算子 |
| **精度验证** | 3 种算子均通过 CPU 参照矩阵转置的 `[ PASS ]` 校验 |
| **性能提升** | 带 Padding 的 Shared Memory 转置带宽明显优于 Naive 版本 |
| **复盘文档** | 在 `docs/day02.md` 中记录 3 种版本的带宽数据与 Bank Conflict 消除推导 |
| **Git 提交** | 完成规范的 Commit 记录：`feat(day02): implement shared memory matrix transpose with padding` |
