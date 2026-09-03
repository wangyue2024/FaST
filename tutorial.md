为期 30 天的高强度冲刺计划以**“硬核代码产出 + 体系结构理论 + 顶会论文精读”**为导向。每天 6 小时的投入按照 **2h 理论/论文 + 3h 动手编码调优 + 1h Profiling/总结** 进行模块化切分。

## 1. 每日 6 小时时间块分配 (Daily Schedule)

| 时间段                   | 模块                        | 核心任务                                                       | 交付标准                                                 |
| :----------------------- | :-------------------------- | :------------------------------------------------------------- | :------------------------------------------------------- |
| **Block 1 (2.0h)** | **体系结构与理论**    | 深入理解 GPU 硬件机理、阅读核心论文（FlashAttention、vLLM 等） | 输出 1 篇结构化论文/技术笔记（包含核心架构图和数据流图） |
| **Block 2 (3.0h)** | **CUDA 核心编码实战** | 算子手撕、C++ 框架搭建、性能调优迭代                           | 提交 Git Commit，代码通过 Google Test 正确性校验         |
| **Block 3 (1.0h)** | **性能分析与复盘**    | 使用 ncu / nsys 分析瓶颈，记录 Roofline 数据与加速比           | 记录当前版本的 FLOPs/带宽利用率，写出下一步优化假设      |

## 2. 4 周进阶学习与开发路线 (4-Week Sprint Roadmap)

### 第 1 周：GPU 硬件微架构与 CUDA 编程范式

* **核心目标：** 搞懂 SM、Warp 调度、内存层级（Global/Shared/Register），掌握基础算子。
* **每日任务：**
  * **Day 1-2：** 精读 *CUDA C++ Programming Guide* 内存与执行模型；手写向量加法与矩阵转置（Matrix Transpose），测试 Coalesced Memory Access（合并访存）对带宽的影响。
  * **Day 3-4：** 深入 Shared Memory 与 Bank Conflict 机制，手写并行规约（Parallel Reduction），完成从 Naive 到 Warp Shuffle（__shfl_down_sync）的 7 级优化。
  * **Day 5-6：** 熟练配置并使用 **NVIDIA Nsight Compute (ncu)**，学会查看 Memory Workload Analysis 和 Roofline Chart。
  * **Day 7：** 周度总结与代码重构，搭建标准化 Benchmark 测试框架（支持 CUDA Event 计时与 CPU 精度校验）。

### 第 2 周：核心战役——从零手撕并极限优化 SGEMM

* **核心目标：** 亲手把单精度矩阵乘法（SGEMM）性能从原生代码推向接近 cuBLAS 的水平。
* **每日任务：**
  * **Day 8-9：** 实现 Naive SGEMM 与 **1D/2D Shared Memory Tiling** 分块，解决数据复用问题。
  * **Day 10-11：** 引入 **Register Tiling**（2D Block Tiling），利用寄存器缓存数据，最大化计算密度并避免 Bank Conflict。
  * **Day 12-13：** 引入 **Vectorized Memory Access**（float4 指令加载）与 **Double Buffering（双缓冲/流水线）**，隐藏全局内存访存延迟。
  * **Day 14：** 对比各版本 GFLOPS 并绘制性能爬升曲线，输出详细的优化复盘文档。

### 第 3 周：算子融合与 FlashAttention 机制剖析

* **核心目标：** 攻克 Memory-Bound 算子，吃透大模型长文本加速的核心技术。
* **每日任务：**
  * **Day 15-16：** 手写 **Safe Softmax** 与 **Online Softmax**（无需保留全量中间结果的一趟扫描算法，FlashAttention 的核心数学基石）。
  * **Day 17-18：** 精读 *FlashAttention-1/2* 论文，在纸上完整推导 Forward Pass 的 Tiling 算法与反向更新公式。
  * **Day 19-21：** 用 CUDA 或 OpenAI Triton 实现一个**简化版 FlashAttention-2 Forward 算子**（支持因果掩码 Causal Masking），在特定矩阵尺寸下与 PyTorch 原生 Attention 对比延迟与显存占用。

### 第 4 周：大模型系统机制、课题组论文与成果打包

* **核心目标：** 拓展至大模型推理系统层面，精准对齐赵杰茹老师/EPCC 课题组的研究前沿。
* **每日任务：**
  * **Day 22-23：** 精读 *PagedAttention (vLLM)* 论文，理解逻辑 Block 到物理 Block 的映射与 KV Cache 显存碎片管理。
  * **Day 24-25：** **课题组精准研读：** 精读赵杰茹老师近期发表的论文（如关于 KV Cache 压缩/检索优化、FPGA 编译框架等），总结其核心贡献与潜在改进点。
  * **Day 26-27：** 整理 GitHub 仓库：编写严谨的 README.md、环境一键配置脚本（Docker/CMake）、架构设计图与 Benchmark 测试图表。
  * **Day 28-30：** 制作一页纸学术简历（CV），撰写联系导师的邮件初稿，准备 15 分钟 PPT 自我陈述。

## 3. 严格的每日与每周检查方法 (Verification & Inspection)

判断自己是否“真正掌握”而不是“伪勤奋”，必须依赖客观指标：

### ① 每日检查闭环（3 步硬性指标）

1. **编译与精度检查：** 新写的 Kernel 必须与 CPU/cuBLAS 结果做差值比较，确保 max_error < 1e-4。
2. **NCU Profiling 检查：** 优化后的算子必须有明确的指标改善（如：*DRAM Throughput 提升*、*Shared Memory Bank Conflict 降为 0* 或 *Compute/Memory Bound 状态发生预期转移*）。
3. **Commit 检查：** 每天必须至少有 1 次有效 Git Commit，包含明确的 Commit Message（如 feat: add double buffering to sgemm, +12% GFLOPS）。

### ② 每周里程碑考核标准

```
[第1周末] 能在 10 分钟内向他人讲清 Warp Divergence 与 Bank Conflict 的物理成因，并给出规避代码。
[第2周末] 手写的 SGEMM 性能达到当前硬件下 cuBLAS 官方库性能的 75% - 85% 以上。
[第3周末] 能白板推导 Online Softmax，并用代码跑通简化版 FlashAttention 前向传播。
[第4周末] GitHub 仓库达到开源标准（有详尽的 Readme、Benchmark 图表、一键复现脚本），CV 准备就绪。
```

## 4. 一个月后的最终交付物 (Final Deliverables)

完成该计划后，你将手握以下直接用于联系导师的硬核筹码：

* **1 个高质量 GitHub 仓库：** 例如命名为 Fast-Ops-CUDA 或 Mini-LLM-Infra，包含手写优化的 SGEMM、Fused Softmax 及 FlashAttention 实现。
* **1 份高质量技术报告/博客：** 详细记录从 Naive 到极致优化的每一步 Profiling 瓶颈分析与解决思路。
* **针对性的 Academic CV：** 重点展示体系结构底座、C++/CUDA 编程能力及对课题组论文的深刻理解
