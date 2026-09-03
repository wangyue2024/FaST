# Day 1 CUDA 冲刺总结报告：GPU 体系结构、内存微架构与多 Kernel 性能对比

---

## 1. 基础环境与工程搭建 (Infrastructure & Engineering)

### 1.1 实验环境与硬件规格
* **GPU 硬件：** NVIDIA GeForce RTX 5060 (8GB VRAM)
* **软件栈：** CUDA Toolkit `12.8` (NVCC V12.8.93), CMake `3.28.3`, GCC `13.3.0`
* **操作系统：** Ubuntu 24.04 LTS (WSL2 / Linux 6.6)

### 1.2 工程结构设计
创建了符合工业规范的模块化工程目录：
```plaintext
/mnt/d/1file/Desktop/code/FaST/
├── include/
│   └── cuda_timer.h     # 公共 CUDA Event 高精度计时器与 CUDA_CHECK 宏
├── src/
│   ├── 01_vector_add/   # Day 1 向量加法与多 Kernel 基准测试
│   └── 02_transpose/    # 矩阵转置对比实验
└── docs/
    └── day01.md         # 本技术复盘报告
```

### 1.3 核心公共组件：`cuda_timer.h`
为了精准测量 GPU 硬件上的运行时间，避免 CPU 发射 Kernel 的异步开销干扰，封装了基于 **CUDA Event** 的高精度计时类 `CudaTimer`：
* **原理：** 直接在 GPU 指令流中插入 `cudaEventRecord` 标记，测量硬件时间戳，精度高达 $0.5 \ \mu\text{s}$。
* **为什么不能直接用 CPU `std::chrono`：** CUDA 内核启动是异步的，CPU 发射后立即返回。若使用 `std::chrono` 必须配合 `cudaDeviceSynchronize()`，但这样会引入额外的 CPU/GPU 线程上下文同步开销。

---

## 2. GPU 体系结构与硬件调度机制 (Hardware Architecture & Scheduling)

### 2.1 硬件层次结构：SM 与 SMSP
* **SM (Streaming Multiprocessor)：** GPU 的核心计算逻辑单元，管理 Block 级别的资源（如 Shared Memory、Block 栅栏同步器）。
* **SMSP (SM Sub-Partition)：** 每个 SM 内部包含 **4 个 SMSP**。SMSP 是独立的硬件指令发射与调度引擎，包含：
  1. 1 个 Warp Scheduler (Warp 调度器) & 1 个 Dispatch Unit
  2. 1 块私有的 Register File 分区
  3. 一组 ALU 执行核心（FP32, INT32, SFU, Tensor Core）

### 2.2 线程一维线性化与 Warp 划分
在 3D Block 内，线程的一维线性序号（Linear Thread ID）计算公式为：
$$\text{Thread ID} = \text{threadIdx.x} + \text{threadIdx.y} \times \text{blockDim.x} + \text{threadIdx.z} \times (\text{blockDim.x} \times \text{blockDim.y})$$

* **关键规律：** **`threadIdx.x` 是变化最快的内层维度**，`threadIdx.z` 是变化最慢的外层维度。
* **Warp 打包逻辑：** GPU 硬件按 Thread ID 从 $0 \sim 31$ 每 32 个线程自动打包为一个 **Warp（线程束）**。因此，`Warp 0` 由连续的 32 个 `threadIdx.x` 组成，这是实现内存合并访问的物理基础。

### 2.3 Block 与 SM 的解耦关系
1. **不可跨 SM 切割：** 1 个 Block 里的所有线程必须完整分配在同一个 SM 上，无法拆分到不同 SM。
2. **一对多挂载：** 1 个 SM 可以同时挂载并并发运行多个 Block（例如 8 个或 16 个），只要 SM 的寄存器、Shared Memory 和 Block 槽位数未触顶。
3. **SMSP 调度 Warp：** 当 SM 挂载多个 Block 时，SM 会将所有 Block 的 Warp 打散并轮询分发给 4 个 SMSP。SMSP 的管理粒度是 **Warp** 而非 Block。

---

## 3. GPU 内存层级与微架构 (Memory Hierarchy & Microarchitecture)

### 3.1 Cacheline 与 Sector 的物理定义
* **Cacheline (128 Bytes)：** L1 Cache 和 L2 Cache 内部**组织与管理**条目的最小数据块，固定为 128 字节。
* **Sector (32 Bytes) 与 Sector Transaction：** SM 与 L2 Cache/DRAM 物理总线之间**实际搬运数据**的最小颗粒度（1 个 Cacheline = 4 个 Sectors）。
* **物理读取规则：** 每次物理传输按 32-Byte（1 个 Sector）为粒度按需传输，避免无谓的带宽浪费。

```plaintext
 Cacheline (128B)  │                         128 Bytes                       │
                   └──────────┬──────────────┬──────────────┬────────────────┘
                              │              │              │
 Sector (32B)      ┌──────────▼───┐┌─────────▼───┐┌─────────▼───┐┌──────────▼───┐
                   │  Sector 0    ││  Sector 1   ││  Sector 2   ││  Sector 3    │  <-- 传输粒度 (32B)
                   └──────────────┘└─────────────┘└─────────────┘└──────────────┘
```

### 3.2 合并访存 (Coalesced) vs 彻底随机访存 (Random Access 8x/16x 惩罚)
* **合并访存 (Coalesced Access)：**
  Warp 32 个线程连续读取 128 字节，刚好触发 4 个 32-Byte Sector Transactions，有效利用率为 $128\text{B}/128\text{B} = \mathbf{100\%}$。
* **随机访存 (Random Access)：**
  32 个线程的访问地址散落在 32 个不同的 Cache Line 中，触发 32 个独立的 32-Byte Sector Transactions（传输 1024 字节），有效利用率仅 $128\text{B}/1024\text{B} = \mathbf{12.5\%}$。实测导致性能**断崖式减速 15.9 倍**。

### 3.3 L2 Cache 击穿现象
* **$1024 \times 1024$ 矩阵 (4 MB)：** 小于 RTX 5060 的 L2 Cache 容量 ($32\text{MB} \sim 64\text{MB}$)，数据全部在 L2 Cache 命中，测得 **`1102.55 GB/s`** 的片上 L2 缓存带宽。
* **$4096 \times 4096$ 矩阵 (64 MB)：** 击穿 L2 Cache，被迫直接读写物理显存 DRAM，测得 **`382.30 GB/s`** 的真实物理显存带宽。

---

## 4. 计算模块与 SFU 硬件微指令 (Compute Engine & Roofline)

### 4.1 Roofline 性能模型与硬件算术强度平衡点
在 RTX 5060 显卡上：
* 物理显存带宽 $\approx 380 \text{ GB/s}$
* FP32 理论算力 $\approx 30 \text{ TFLOPS}$
* **算术强度平衡点：**
  $$\text{Break-even Point} = \frac{30 \times 10^{12} \text{ FLOPs/s}}{3.8 \times 10^{11} \text{ Bytes/s}} \approx \mathbf{79 \text{ FLOPs / Byte}}$$
  **结论：读写 1 个 4-Byte `float` 的显存开销，足够 ALU 执行 316 次 FP32 浮点运算。** 当每元素计算次数未超过 300 次时，算子绝对处于 Memory-Bound 状态。

### 4.2 SFU (Special Function Units) 与高级 C++ 优化技巧
每个 SMSP 拥有 4 个 SFU，内部通过 ROM 查找表 + 二次多项式插值电路实现 $2 \sim 4$ 周期极速超越函数计算：
1. **`__sincosf(a, &sa, &ca)` 硬件融合微指令：** 一次性同时算出 $\sin$ 和 $\cos$，正余弦计算开销直接降低 50%。
2. **`exp2f(a * COEFF)` 底数转换与常数折叠：** 将自然指数 $e^x$ 转换为 $2^{x \cdot \log_2(e)}$，配合 `constexpr float COEFF` 在编译期预先折叠。
3. **`#pragma unroll` 循环展开：** 消除循环控制指令与跳转开销，提升指令级并行（ILP）。
4. **`__restrict__` 指针别名提示：** 消除内存重叠假设，触发 NVCC 生成只读 `LDG` 指令。
5. **`-use_fast_math` 编译标志：** 全局将标准 IEEE 算术函数替换为 SFU 硬件极速微指令。

---

## 5. 多 Kernel 实测综合对比数据 (Benchmark Results)

在 $4096 \times 4096$ 矩阵（单矩阵 64 MB，超越 L2 Cache）下，5 种典型模式的对比数据如下：

| 序号 | 模式名称 (Kernel Execution Mode) | 平均耗时 (ms) | 有效带宽 (GB/s) | 相对减速比 (Slowdown) | 核心瓶颈与物理机制 |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | **Row-Major Coalesced Add (Baseline)** | **`0.528 ms`** | **`381.07 GB/s`** | **`1.000x`** | 纯合并访存，达到物理显存带宽极限 |
| **2** | **Col-Major Non-Coalesced Add** | **`0.741 ms`** | **`271.78 GB/s`** | **`1.402x`** | 按列跨步访问，受 L2 Cache 邻近预取缓解 |
| **3** | **Naive Matrix Transpose** | **`0.387 ms`** | **`347.25 GB/s`** | **`0.732x`** | 读连续写跨步（单矩阵传输，耗时更短） |
| **4** | **Complex Floating Compute (Standard)** | **`4.641 ms`** | **`43.38 GB/s`** | **`8.151x`** | IEEE 标准函数，切换为 Compute-Bound |
| **4.5**| **Complex Compute (SFU + sincos + unroll)**| **`1.770 ms`** | **`113.76 GB/s`** | **`3.108x`** | **比 Mode 4 加速 2.62x！** 充分释放 SFU 算力 |
| **5** | **Pure Random Access (L2 Cache Destroyed)**| **`8.730 ms`** | **`30.75 GB/s`** | **`16.523x`** | **崩溃式减速 16.5x！** 8x Sector 膨胀 $\times$ 1.33x 索引开销 $\times$ 1.5x 间接寻址串行化 |

---

## 6. 复盘总结与 Day 2 展望 (Conclusion & Outlook)

### 6.1 Day 1 核心收获
1. **代码实践：** 搭建了高精度 [cuda_timer.h](file:///mnt/d/1file/Desktop/code/FaST/include/cuda_timer.h)，完成了 [main.cu](file:///mnt/d/1file/Desktop/code/FaST/src/01_vector_add/main.cu) 与 [blockDim2Sequence.cu](file:///mnt/d/1file/Desktop/code/FaST/src/01_vector_add/blockDim2Sequence.cu) 完整性能基准测试。
2. **理论验证：** 实测验证了合并访存对显存带宽的决定性作用，推导并验证了 L2 Cache 击穿与纯随机访存 16 倍减速的底座逻辑。

### 6.2 Day 2 冲刺预告
* ** Shared Memory（片上 SRAM 共享内存）** 的开辟与使用
* 解决矩阵转置中的非合并写瓶颈
* **破解 32-Bank Conflict（存储块冲突）** 与 Padding 避让技巧
