启动冲刺的第一步是 **在 30 分钟内跑通基础设施** ，避免把时间浪费在配环境的内耗中。

### 0. 启动前准备（30分钟基建）

* **创建 GitHub 仓库：** 命名为 `Fast-CUDA-Ops` 或 `CUDA-Sprint-30D`，初始化 `CMakeLists.txt`、`.gitignore` 和 `README.md`。
* **确认工具链：** 终端验证 `nvcc --version`、`ncu --version` 可用；确保机器具备 CUDA Toolkit **$\ge$** 12.0 及 CMake **$\ge$** 3.20。
* **准备目录结构：**

  **Plaintext**

  ```
  ├── CMakeLists.txt
  ├── include/          # 公共头文件、CUDA 计时宏、CPU 验证函数
  ├── src/
  │   ├── 01_vector_add/
  │   └── 02_transpose/
  └── profile/          # 存放 ncu 分析报告与截图
  ```

### Day 1 落地执行方案（按 6 小时模块切分）

#### Block 1：理论与体系结构（2.0h）

* **核心输入：**

  * 精读 *CUDA C++ Programming Guide* 中 **Programming Model** 和 **Hardware Implementation** 章节。
  * 重点搞懂：SM（Streaming Multiprocessor）、Warp（32 线程调度单位）、SIMT 执行模型、Global Memory 的合并访存（Coalesced Memory Access）机制。
* **交付笔记内容：**

  * 画出 1 张 Warp 调度与 Memory Coalescing 机制图。
  * 用公式推导：当访存步长（Stride）从 1 变成 2、4、32 时，L1/L2 缓存事务（Memory Transactions）的变化倍数。

#### Block 2：核心编码实战（3.0h）

* **任务 1：Vector Add 基准搭建（1.0h）**

  * 编写 `vector_add_cpu` 与 `vector_add_gpu`。
  * 引入 CUDA Event 计时与 CPU 精度校验（断言 `|gpu[i] - cpu[i]| < 1e-5`）。
* **任务 2：Matrix Transpose 对比实验（2.0h）**

  * **版本 A（Coalesced Write, Non-Coalesced Read）：** 转置读取跨步，写入连续。
  * **版本 B（Coalesced Read, Non-Coalesced Write）：** 读取连续，写入跨步。
  * **版本 C（Naive Copy）：** 纯拷贝作为带宽理论上限 Baseline。
* **交付代码：** 通过编译并运行不同矩阵尺寸（如 **$4096 \times 4096$**）的耗时对比。

#### Block 3：性能分析与复盘（1.0h）

* **NCU 实测：**

  * 运行命令行抓取分析：

    **Bash**

    ```
    ncu --set full -o profile/transpose_naive src/02_transpose/transpose_bench
    ```
  * 对比版本 A 与版本 B 的 `DRAM Bandwidth`（显存带宽利用率）与 `Sector Reads/Writes` 指标。
* **Git 提交与复盘记录：**

  * 提交 Commit：`feat(day01): implement vector_add and transpose naive kernels with ncu profiling`。
  * 在仓库 `docs/day01.md` 记录：矩阵转置非合并访存造成的带宽惩罚（通常有 2x-5x 延迟差异）。

你当前使用的显卡型号是什么？如果本地开发环境（如 `nvcc` 或 `ncu`）尚未配好，可以先确认 GPU 架构（如 Ampere / Ada / Hopper），以便在 CMakeLists 中指定对应的 `CUDA_ARCH`。
