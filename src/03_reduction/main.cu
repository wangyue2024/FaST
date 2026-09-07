#include "../../include/cuda_timer.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

// =========================================================================================
// 1. v1: 朴素交错归约 (Naive Divergent)
// 问题分析：if (tid % (2 * i) == 0) 导致严重 Warp Divergence（线程束分化）
// 修复点：补全循环内部 missing 的 __syncthreads() 屏障同步
// =========================================================================================
__global__ void kernel_reduce_v1_divergent(const float *__restrict__ in,
                                           float *__restrict__ out, int N) {
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = (idx < N) ? in[idx] : 0.0f;
  __syncthreads();

  for (int i = 1; i < blockDim.x; i <<= 1) {
    if (tid % (2 * i) == 0) {
      sdata[tid] += sdata[tid + i];
    }
    __syncthreads(); // ✅ 必须加上屏障同步，防止数据竞争
  }

  if (tid == 0) {
    atomicAdd(out, sdata[0]);
  }
}

// =========================================================================================
// 2. v2: 连续线程归约 (Sequential Threads)
// 优化：使活跃线程连续 (tid < blockDim.x / 2)，消除 Warp Divergence
// 问题分析：int index = 2 * i * tid 导致 16-way Shared Memory Bank Conflict
// =========================================================================================
__global__ void kernel_reduce_v2_sequential(const float *__restrict__ in,
                                            float *__restrict__ out, int N) {
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = (idx < N) ? in[idx] : 0.0f;
  __syncthreads();

  for (int i = 1; i < blockDim.x; i <<= 1) {
    int index = 2 * i * tid;
    if (index < blockDim.x) {
      sdata[index] += sdata[index + i]; // 存在 Bank Conflict
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(out, sdata[0]);
  }
}

// =========================================================================================
// 3. v3: 反向交错步长归约 (Interleaved Addressing)
// 优化：反向步长 s = blockDim.x / 2，连续线程读取连续 Shared Memory
// 地址，彻底消除 Bank Conflict
// =========================================================================================
__global__ void kernel_reduce_v3_interleaved(const float *__restrict__ in,
                                             float *__restrict__ out, int N) {
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = (idx < N) ? in[idx] : 0.0f;
  __syncthreads();

  for (int i = blockDim.x / 2; i > 0; i >>= 1) {
    if (tid < i) {
      sdata[tid] +=
          sdata[tid + i]; // ✅ 无 Bank Conflict，连续线程读写连续 Shared Mem
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(out, sdata[0]);
  }
}

// =========================================================================================
// 4. v4: Load 阶段首加 (First Add on Load / Grid-Stride Loop)
// 优化：从 Global Memory 载入 Shared Memory
// 时先完成加法，数据搬运量减半，Shared Mem 占用减半
// =========================================================================================
__global__ void kernel_reduce_v4_first_add(const float *__restrict__ in,
                                           float *__restrict__ out, int N) {
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

  // 网格跨步读取并完成第一轮累加
  float sum = 0.0f;
  if (idx < N)
    sum += in[idx];
  if (idx + blockDim.x < N)
    sum += in[idx + blockDim.x];
  sdata[tid] = sum;
  __syncthreads();

  for (int i = blockDim.x / 2; i > 0; i >>= 1) {
    if (tid < i) {
      sdata[tid] += sdata[tid + i];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(out, sdata[0]);
  }
}

// =========================================================================================
// 5. v5: 展开最后一个 Warp (Unroll Last Warp)
// 优化：当 i <= 32 时进入最后一个 Warp，使用 __shfl_down_sync 消除屏障与 UB
// 风险
// =========================================================================================
__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

__global__ void kernel_reduce_v5_unroll_last(const float *__restrict__ in,
                                             float *__restrict__ out, int N) {
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

  float sum = 0.0f;
  if (idx < N)
    sum += in[idx];
  if (idx + blockDim.x < N)
    sum += in[idx + blockDim.x];
  sdata[tid] = sum;
  __syncthreads();

  // Block 级树状归约只归约到前 64 个元素 (即 i > 32 停止)
  for (int i = blockDim.x / 2; i > 32; i >>= 1) {
    if (tid < i) {
      sdata[tid] += sdata[tid + i];
    }
    __syncthreads();
  }

  // 最后一个 Warp (前 32 个线程) 处理剩余的 64 个元素 (sdata[tid] + sdata[tid +
  // 32])
  if (tid < 32) {
    float val = sdata[tid] + sdata[tid + 32]; // ✅ 加上后半部分 32 个元素
    val = warp_reduce_sum(val);
    if (tid == 0) {
      atomicAdd(out, val);
    }
  }
}

// =========================================================================================
// 6. v6: 极致 Warp Shuffle 归约 (Zero Shared Memory For Tree Reduction)
// 优化：网格跨步网格累加 + Warp 级 Shuffle 归约 + Block 级汇总，完全摆脱 Shared
// Memory 树状归约
// =========================================================================================
__global__ void kernel_reduce_v6_shuffle(const float *__restrict__ in,
                                         float *__restrict__ out, int N) {
  // 网格跨步循环 (Grid-Stride Loop)，每个线程累加多个数据点
  float sum = 0.0f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    sum += in[i];
  }

  // 1. Warp 内部寄存器归约
  sum = warp_reduce_sum(sum);

  int laneId = threadIdx.x % 32;
  int warpId = threadIdx.x / 32;

  // 2. 每个 Warp 的 0 号线程将结果写入 Shared Memory
  __shared__ float warp_sums[32];
  if (laneId == 0) {
    warp_sums[warpId] = sum;
  }
  __syncthreads();

  // 3. 由第 0 个 Warp 完成 Block 级最终归约
  sum = (threadIdx.x < (blockDim.x / 32)) ? warp_sums[laneId] : 0.0f;
  if (warpId == 0) {
    sum = warp_reduce_sum(sum);
    if (threadIdx.x == 0) {
      atomicAdd(out, sum);
    }
  }
}

// =========================================================================================
// 7. v7: 终极融合归约 (First-Add Coalesced Load + Warp Shuffle Reduction)
// 优化原理：
// 1. 采用 v5 的 1KB Coalesced 首加读取，规避 8MB 巨型 Grid-Stride 跨步引起的 L2
// Cache / DRAM Page 冲突
// 2. 采用 v6 的 Warp Shuffle 寄存器归约，摆脱 Shared Memory 树状屏障同步依赖
// 3. 大 Grid 规模 (blocks / 2 = 32768) 产生满载 DRAM 并发吞吐量 (MLP)
// =========================================================================================
__global__ void kernel_reduce_v7_ultimate(const float *__restrict__ in,
                                          float *__restrict__ out, int N) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

  float sum = 0.0f;
  if (idx < N)
    sum += in[idx];
  if (idx + blockDim.x < N)
    sum += in[idx + blockDim.x];

  // 1. Warp 内部寄存器归约
  sum = warp_reduce_sum(sum);

  int laneId = tid % 32;
  int warpId = tid / 32;

  // 2. 每个 Warp 的 0 号线程写入 Shared Memory
  __shared__ float warp_sums[32];
  if (laneId == 0) {
    warp_sums[warpId] = sum;
  }
  __syncthreads();

  // 3. 第 0 个 Warp 完成 Block 级汇总归约
  if (warpId == 0) {
    sum = (tid < (blockDim.x / 32)) ? warp_sums[laneId] : 0.0f;
    sum = warp_reduce_sum(sum);
    if (tid == 0) {
      atomicAdd(out, sum);
    }
  }
}

// =========================================================================================
// 8. v8: 多累加器指令重排 + Warp Shuffle 归约 (Multi-Accumulator ILP + Shuffle)
// 优化原理：
// 1. 在 v6 网格跨步循环的基础上，使用 4 个独立的累加器寄存器 (sum0, sum1, sum2,
// sum3)
// 2. 彻底消除 RAW (Read-After-Write) 数据依赖，迫使 NVCC 连续发射 4 条 LDG.E
// 指令
// 3. 4 条 DRAM 读请求并发流水线化 (MLP 满载)，随后做 Warp Shuffle 寄存器归约
// =========================================================================================
__global__ void kernel_reduce_v8_ilp_shuffle(const float *__restrict__ in,
                                             float *__restrict__ out, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

  for (int i = tid; i < N; i += stride * 4) {
    float v0 = (i < N) ? in[i] : 0.0f;
    float v1 = (i + stride < N) ? in[i + stride] : 0.0f;
    float v2 = (i + stride * 2 < N) ? in[i + stride * 2] : 0.0f;
    float v3 = (i + stride * 3 < N) ? in[i + stride * 3] : 0.0f;

    sum0 += v0;
    sum1 += v1;
    sum2 += v2;
    sum3 += v3;
  }

  float sum = sum0 + sum1 + sum2 + sum3;

  // 1. Warp 内部寄存器归约
  sum = warp_reduce_sum(sum);

  int laneId = threadIdx.x % 32;
  int warpId = threadIdx.x / 32;

  // 2. 每个 Warp 的 0 号线程写入 Shared Memory
  __shared__ float warp_sums[32];
  if (laneId == 0) {
    warp_sums[warpId] = sum;
  }
  __syncthreads();

  // 3. 第 0 个 Warp 完成 Block 级汇总归约
  if (warpId == 0) {
    sum = (threadIdx.x < (blockDim.x / 32)) ? warp_sums[laneId] : 0.0f;
    sum = warp_reduce_sum(sum);
    if (threadIdx.x == 0) {
      atomicAdd(out, sum);
    }
  }
}

// =========================================================================================
// 9. v9: Block 连续切片读取 + ILP + Shuffle 归约 (Block-Contiguous Tile
// Reading) 优化原理：
// 1. 将 16M 数据平分成 gridDim.x 块连续 Tile，每个 Block 独占一小块连续内存
// (15KB ~ 64KB)
// 2. Block 内部做相隔仅仅 1KB (blockDim.x * 4) 的连续 Tile 循环，100% 命中 DRAM
// Row Buffer
// 3. 兼具了 v5 的 100% DRAM 物理空间局部性 与 v6/v8 固定 1088 Block
// 的任意尺寸通用性
// =========================================================================================
__global__ void kernel_reduce_v9_block_contiguous(const float *__restrict__ in,
                                                  float *__restrict__ out,
                                                  int N) {
  int total_threads = blockDim.x * gridDim.x;
  int items_per_thread = (N + total_threads - 1) / total_threads;
  int items_per_block = items_per_thread * blockDim.x;

  int block_offset = blockIdx.x * items_per_block;
  int block_end = min(block_offset + items_per_block, N);

  float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
  int step = blockDim.x * 4;

  for (int i = block_offset + threadIdx.x; i < block_end; i += step) {
    float v0 = (i < block_end) ? in[i] : 0.0f;
    float v1 = (i + blockDim.x < block_end) ? in[i + blockDim.x] : 0.0f;
    float v2 = (i + blockDim.x * 2 < block_end) ? in[i + blockDim.x * 2] : 0.0f;
    float v3 = (i + blockDim.x * 3 < block_end) ? in[i + blockDim.x * 3] : 0.0f;

    sum0 += v0;
    sum1 += v1;
    sum2 += v2;
    sum3 += v3;
  }

  float sum = sum0 + sum1 + sum2 + sum3;

  // 1. Warp 内部寄存器归约
  sum = warp_reduce_sum(sum);

  int laneId = threadIdx.x % 32;
  int warpId = threadIdx.x / 32;

  // 2. 每个 Warp 的 0 号线程写入 Shared Memory
  __shared__ float warp_sums[32];
  if (laneId == 0) {
    warp_sums[warpId] = sum;
  }
  __syncthreads();

  // 3. 第 0 个 Warp 完成 Block 级汇总归约
  if (warpId == 0) {
    sum = (threadIdx.x < (blockDim.x / 32)) ? warp_sums[laneId] : 0.0f;
    sum = warp_reduce_sum(sum);
    if (threadIdx.x == 0) {
      atomicAdd(out, sum);
    }
  }
}

// =========================================================================================
// 10. v10: float4 向量化 (128-bit LDG.128) + Warp Shuffle 归约
// 优化原理：
// 1. 将 float* 强转为 float4*，一条显存指令直接发射 128-bit (16 字节) 向量读取
// 2. 将指令发射开销减少 75%，极大地提升 SM 指令译码吞吞吞吐量
// 3. 将每个 float4 元素在寄存器内解包求和，随后接 Warp Shuffle 寄存器归约
// =========================================================================================
__global__ void kernel_reduce_v10_vectorized_float4(const float * __restrict__ in, float * __restrict__ out, int N) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  const float4* in_v4 = reinterpret_cast<const float4*>(in);
  int vec_N = N / 4;

  float sum = 0.0f;
  if (idx < vec_N) {
    float4 v = in_v4[idx];
    sum += v.x + v.y + v.z + v.w;
  }
  if (idx + blockDim.x < vec_N) {
    float4 v = in_v4[idx + blockDim.x];
    sum += v.x + v.y + v.z + v.w;
  }

  // 1. Warp 内部寄存器归约
  sum = warp_reduce_sum(sum);

  int laneId = tid % 32;
  int warpId = tid / 32;

  // 2. Shared Memory 汇总各 Warp 结果
  __shared__ float warp_sums[32];
  if (laneId == 0) {
    warp_sums[warpId] = sum;
  }
  __syncthreads();

  // 3. 第 0 个 Warp 完成 Block 级汇总归约
  if (warpId == 0) {
    sum = (tid < (blockDim.x / 32)) ? warp_sums[laneId] : 0.0f;
    sum = warp_reduce_sum(sum);
    if (tid == 0) {
      atomicAdd(out, sum);
    }
  }
}


// CPU 基准归约计算 (Double 精度防累加误差)
double reduce_cpu(const float *data, int N) {
  double sum = 0.0;
  for (int i = 0; i < N; ++i) {
    sum += static_cast<double>(data[i]);
  }
  return sum;
}

// 相对误差比对
bool verify_result(double cpu_sum, float gpu_sum) {
  double diff = std::abs(cpu_sum - static_cast<double>(gpu_sum));
  double rel_err = diff / std::abs(cpu_sum);
  if (rel_err > 1e-4) {
    std::cerr << "Verification failed! CPU=" << cpu_sum << ", GPU=" << gpu_sum
              << ", RelErr=" << rel_err << std::endl;
    return false;
  }
  return true;
}

struct TestResult {
  std::string name;
  float avg_time_ms;
  double bandwidth_gbps;
  float speedup;
};

int main() {
  const int N = 16777216; // 16M float (64 MB)
  const size_t bytes = N * sizeof(float);

  std::cout << "==============================================================="
               "==========================\n";
  std::cout
      << "     CUDA Parallel Reduction Benchmark (1000 Repetitions, Elements: "
      << N << " / 64MB)\n";
  std::cout << "==============================================================="
               "==========================\n\n";

  std::vector<float> h_in(N);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (int i = 0; i < N; ++i) {
    h_in[i] = dist(rng);
  }

  std::cout << "Calculating CPU Reference Sum (Double Precision)..."
            << std::flush;
  double cpu_sum = reduce_cpu(h_in.data(), N);
  std::cout << " [ Done: " << std::fixed << std::setprecision(4) << cpu_sum
            << " ]\n\n";

  float *d_in = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

  CudaTimer timer;
  const int warmup_iters = 10;
  const int num_iters = 1000;

  std::vector<TestResult> results;
  float baseline_time = 0.0f;

  auto run_test = [&](const std::string &name, int block_size, int grid_size,
                      int shared_mem_bytes, auto kernel_launch) {
    // 正确性校验
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
    kernel_launch(grid_size, block_size, shared_mem_bytes);
    CUDA_CHECK(cudaDeviceSynchronize());
    float h_out = 0.0f;
    CUDA_CHECK(
        cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    bool passed = verify_result(cpu_sum, h_out);

    // 预热
    for (int i = 0; i < warmup_iters; ++i) {
      CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
      kernel_launch(grid_size, block_size, shared_mem_bytes);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // 1000 次纯 GPU 计时
    timer.start();
    for (int i = 0; i < num_iters; ++i) {
      kernel_launch(grid_size, block_size, shared_mem_bytes);
    }
    float avg_time_ms = timer.stop() / num_iters;

    // 单次读取 N * 4 字节的数据
    double bandwidth_gbps = (bytes) / (avg_time_ms * 1e6);

    if (baseline_time == 0.0f) {
      baseline_time = avg_time_ms;
    }
    float speedup = baseline_time / avg_time_ms;

    results.push_back({name, avg_time_ms, bandwidth_gbps, speedup});

    std::cout << std::left << std::setw(48) << name << " Verification: ["
              << (passed ? " PASS " : " FAIL ") << "]" << std::endl;
  };

  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;

  // 1. v1 Divergent
  run_test("1. Reduce v1 (Naive Divergent)", threads, blocks,
           threads * sizeof(float), [&](int g, int b, int sm) {
             kernel_reduce_v1_divergent<<<g, b, sm>>>(d_in, d_out, N);
           });

  // 2. v2 Sequential
  run_test("2. Reduce v2 (Sequential Threads)", threads, blocks,
           threads * sizeof(float), [&](int g, int b, int sm) {
             kernel_reduce_v2_sequential<<<g, b, sm>>>(d_in, d_out, N);
           });

  // 3. v3 Interleaved
  run_test("3. Reduce v3 (Interleaved Addressing)", threads, blocks,
           threads * sizeof(float), [&](int g, int b, int sm) {
             kernel_reduce_v3_interleaved<<<g, b, sm>>>(d_in, d_out, N);
           });

  // 4. v4 First Add on Load
  run_test("4. Reduce v4 (First Add on Load)", threads, blocks / 2,
           threads * sizeof(float), [&](int g, int b, int sm) {
             kernel_reduce_v4_first_add<<<g, b, sm>>>(d_in, d_out, N);
           });

  // 5. v5 Unroll Last Warp
  run_test("5. Reduce v5 (Unroll Last Warp)", threads, blocks / 2,
           threads * sizeof(float), [&](int g, int b, int sm) {
             kernel_reduce_v5_unroll_last<<<g, b, sm>>>(d_in, d_out, N);
           });

  // 6. v6 Warp Shuffle
  int sm_count = 0;
  CUDA_CHECK(
      cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
  int grid_shuffle = sm_count * 32;
  run_test("6. Reduce v6 (Warp Shuffle Extreme)", threads, grid_shuffle, 0,
           [&](int g, int b, int sm) {
             kernel_reduce_v6_shuffle<<<g, b>>>(d_in, d_out, N);
           });

  // 7. v7 Ultimate Hybrid
  run_test("7. Reduce v7 (Ultimate Hybrid)", threads, blocks / 2, 0,
           [&](int g, int b, int sm) {
             kernel_reduce_v7_ultimate<<<g, b>>>(d_in, d_out, N);
           });

  // 8. v8 ILP Reorder Shuffle
  run_test("8. Reduce v8 (ILP Reorder Shuffle)", threads, grid_shuffle, 0,
           [&](int g, int b, int sm) {
             kernel_reduce_v8_ilp_shuffle<<<g, b>>>(d_in, d_out, N);
           });

  // 9. v9 Block-Contiguous Tile
  run_test("9. Reduce v9 (Block-Contiguous Tile)", threads, grid_shuffle, 0,
           [&](int g, int b, int sm) {
             kernel_reduce_v9_block_contiguous<<<g, b>>>(d_in, d_out, N);
           });

  // 10. v10 Vectorized float4
  int vec_N = N / 4;
  int blocks_v10 = (vec_N + (threads * 2) - 1) / (threads * 2);
  run_test("10. Reduce v10 (Vectorized float4)", threads, blocks_v10, 0,
           [&](int g, int b, int sm) {
             kernel_reduce_v10_vectorized_float4<<<g, b>>>(d_in, d_out, N);
           });


  // 输出性能比对表格
  std::cout << "\n============================================================="
               "============================\n";
  std::cout << std::left << std::setw(46) << "Kernel Variant" << std::setw(15)
            << "Avg Time (ms)" << std::setw(20) << "Bandwidth (GB/s)"
            << std::setw(12) << "Speedup" << "\n";
  std::cout << "---------------------------------------------------------------"
               "--------------------------\n";

  for (const auto &res : results) {
    std::cout << std::left << std::setw(46) << res.name << std::fixed
              << std::setprecision(3) << std::setw(15) << res.avg_time_ms
              << std::setw(20) << res.bandwidth_gbps << std::setprecision(2)
              << res.speedup << "x\n";
  }
  std::cout << "==============================================================="
               "==========================\n\n";

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));

  return 0;
}