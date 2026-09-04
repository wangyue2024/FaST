#include "../../include/cuda_timer.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// 1a. 朴素转置 (Read Coalesced, Write Stride: 读合并，写跨步)
__global__ void kernel_naive_transpose(const float *A, float *C, int W, int H) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < W && y < H) {
    C[x * H + y] = A[y * W + x];
  }
}

// 1b. 朴素转置 交换 x/y (Read Stride, Write Coalesced: 读跨步，写合并)
__global__ void kernel_naive_transpose_swapped(const float *A, float *C, int W,
                                               int H) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < W && y < H) {
    C[y * W + x] = A[x * H + y];
  }
}

// 2a. 冗余 Tiling 原始顺序版本（外层 i, 内层 j -> 单线程写连续 C[idx*H+idy]）
__global__ void kernel_tiling_no_shared_transpose(const float *A, float *C,
                                                  int W, int H, int TILE_W,
                                                  int TILE_H) {
  int tile_x = (blockIdx.x * blockDim.x + threadIdx.x) * TILE_W;
  int tile_y = (blockIdx.y * blockDim.y + threadIdx.y) * TILE_H;
  for (int i = 0; i < TILE_W; i++) {
    for (int j = 0; j < TILE_H; ++j) {
      int idx = tile_x + i;
      int idy = tile_y + j;
      if (idx < W && idy < H) {
        C[idx * H + idy] = A[idy * W + idx];
      }
    }
  }
}

// 2b. 冗余 Tiling 调换循环顺序版本（外层 j, 内层 i -> 单线程读连续 A[idy*W+idx]）
__global__ void kernel_tiling_no_shared_transpose_swapped(const float *A,
                                                          float *C, int W,
                                                          int H, int TILE_W,
                                                          int TILE_H) {
  int tile_x = (blockIdx.x * blockDim.x + threadIdx.x) * TILE_W;
  int tile_y = (blockIdx.y * blockDim.y + threadIdx.y) * TILE_H;
  for (int j = 0; j < TILE_H; j++) {
    for (int i = 0; i < TILE_W; ++i) {
      int idx = tile_x + i;
      int idy = tile_y + j;
      if (idx < W && idy < H) {
        C[idx * H + idy] = A[idy * W + idx];
      }
    }
  }
}

// 3. 基础 Shared Memory 转置 (32x32, 存在 32-way Bank Conflict)
__global__ void kernel_shared_memory_transpose(const float *A, float *C, int W,
                                               int H) {
  __shared__ float tile_shared[32][32];

  int idx = blockIdx.x * 32 + threadIdx.x;
  int idy = blockIdx.y * 32 + threadIdx.y;
  if (idx < W && idy < H) {
    tile_shared[threadIdx.y][threadIdx.x] = A[idy * W + idx];
  }
  __syncthreads();

  int outx = blockIdx.y * 32 + threadIdx.x;
  int outy = blockIdx.x * 32 + threadIdx.y;
  if (outx < H && outy < W) {
    C[outy * H + outx] = tile_shared[threadIdx.x][threadIdx.y];
  }
}

// 4. 带 Padding 的 Shared Memory 转置 (32x33, 无 Bank Conflict)
__global__ void kernel_shared_memory_padding_transpose(const float *A, float *C,
                                                       int W, int H) {
  __shared__ float tile_shared[32][33];

  int idx = blockIdx.x * 32 + threadIdx.x;
  int idy = blockIdx.y * 32 + threadIdx.y;
  if (idx < W && idy < H) {
    tile_shared[threadIdx.y][threadIdx.x] = A[idy * W + idx];
  }
  __syncthreads();

  int outx = blockIdx.y * 32 + threadIdx.x;
  int outy = blockIdx.x * 32 + threadIdx.y;
  if (outx < H && outy < W) {
    C[outy * H + outx] = tile_shared[threadIdx.x][threadIdx.y];
  }
}

// 5. 线程 Tile 化 + Padding 转置
__global__ void kernel_shared_memory_tile_padding_transpose(const float *A,
                                                            float *C, int W,
                                                            int H) {
  __shared__ float tile_shared[32][33];

  int idx = blockIdx.x * 32 + threadIdx.x;
  int idy = blockIdx.y * 32 + threadIdx.y;

  if (idx < W) {
#pragma unroll
    for (int i = 0; i < 32; i += blockDim.y) {
      if (idy + i < H) {
        tile_shared[threadIdx.y + i][threadIdx.x] = A[(idy + i) * W + idx];
      }
    }
  }
  __syncthreads();

  int outx = blockIdx.y * 32 + threadIdx.x;
  int outy = blockIdx.x * 32 + threadIdx.y;

  if (outx < H) {
#pragma unroll
    for (int i = 0; i < 32; i += blockDim.y) {
      if (outy + i < W) {
        C[(outy + i) * H + outx] = tile_shared[threadIdx.x][threadIdx.y + i];
      }
    }
  }
}

// 6. 工业级最标准最高效标准算子 (Production-Grade Standard Transpose)
constexpr int TILE_DIM = 32;
constexpr int BLOCK_ROWS = 8;

__global__ void __launch_bounds__(256)
kernel_transpose_production_optimized(const float *__restrict__ A,
                                       float *__restrict__ C, int W, int H) {
  __shared__ float tile[TILE_DIM][TILE_DIM + 1];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;

#pragma unroll
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    if (x < W && (y + j) < H) {
      tile[threadIdx.y + j][threadIdx.x] = A[(y + j) * W + x];
    }
  }

  __syncthreads();

  int out_x = blockIdx.y * TILE_DIM + threadIdx.x;
  int out_y = blockIdx.x * TILE_DIM + threadIdx.y;

#pragma unroll
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    if (out_x < H && (out_y + j) < W) {
      C[(out_y + j) * H + out_x] = tile[threadIdx.x][threadIdx.y + j];
    }
  }
}

// 7. 极致向量化转置算子 (float4 Extreme Transpose - 128-bit LDG/STG)
__global__ void kernel_transpose_float4_extreme(const float *__restrict__ in,
                                                float *__restrict__ out,
                                                int width, int height) {
  // 行步长设为 32 + 4 = 36，避开 4 字节 32-Bank 冲突
  __shared__ float s_tile[32][36];

  int tid_x = threadIdx.x; // 0 ~ 7
  int tid_y = threadIdx.y; // 0 ~ 7

  int bx = blockIdx.x * 32;
  int by = blockIdx.y * 32;

  // 1. 向量化全局内存读取 (float4 128-bit LDG) 并解包进 Shared Memory
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int row = tid_y + i * 8;
    int col = tid_x * 4;

    const float4 *in_ptr = reinterpret_cast<const float4 *>(
        in + (by + row) * width + (bx + col));
    float4 v = *in_ptr;

    s_tile[row][col + 0] = v.x;
    s_tile[row][col + 1] = v.y;
    s_tile[row][col + 2] = v.z;
    s_tile[row][col + 3] = v.w;
  }

  __syncthreads();

  int out_bx = blockIdx.y * 32;
  int out_by = blockIdx.x * 32;

  // 2. 从 Shared Memory 竖向按列收集 4 个连续数据拼装成 float4，再用 128-bit 指令写回 DRAM
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int row = tid_y + i * 8;
    int col = tid_x * 4;

    float4 v;
    v.x = s_tile[col + 0][row];
    v.y = s_tile[col + 1][row];
    v.z = s_tile[col + 2][row];
    v.w = s_tile[col + 3][row];

    float4 *out_ptr = reinterpret_cast<float4 *>(
        out + (out_by + row) * height + (out_bx + col));
    *out_ptr = v;
  }
}

struct TestResult {
  std::string name;
  int threads_per_block;
  int elem_per_thread;
  float avg_time_ms;
  double bandwidth_gbps;
  float speedup;
};

int main() {
  const int W = 4096;
  const int H = 4096;
  const size_t total_elements = static_cast<size_t>(W) * H;
  const size_t bytes = total_elements * sizeof(float);

  std::cout << "==============================================================="
               "==========================\n";
  std::cout << "    CUDA Matrix Transpose Ultra-Stable Benchmark (1000 "
               "Repetitions, Size: "
            << W << " x " << H << ")\n";
  std::cout << "==============================================================="
               "==========================\n\n";

  std::vector<float> h_A(total_elements);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (size_t i = 0; i < total_elements; ++i) {
    h_A[i] = dist(rng);
  }

  float *d_A = nullptr, *d_C = nullptr;
  CUDA_CHECK(cudaMalloc(&d_A, bytes));
  CUDA_CHECK(cudaMalloc(&d_C, bytes));
  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));

  CudaTimer timer;
  const int warmup_iters = 20;
  const int num_iters = 1000;

  std::vector<TestResult> results;
  float baseline_time = 0.0f;

  auto run_test = [&](const std::string &name, int threads_per_block,
                      int elem_per_thread, dim3 grid, dim3 block,
                      auto kernel_func) {
    CUDA_CHECK(cudaMemset(d_C, 0, bytes));

    for (int i = 0; i < warmup_iters; ++i) {
      kernel_func(grid, block);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    timer.start();
    for (int i = 0; i < num_iters; ++i) {
      kernel_func(grid, block);
    }
    float total_time_ms = timer.stop();
    float avg_time_ms = total_time_ms / num_iters;

    double bandwidth_gbps = (2.0 * bytes) / (avg_time_ms * 1e6);

    if (baseline_time == 0.0f) {
      baseline_time = avg_time_ms;
    }
    float speedup = baseline_time / avg_time_ms;

    results.push_back({name, threads_per_block, elem_per_thread, avg_time_ms,
                       bandwidth_gbps, speedup});

    std::cout << std::left << std::setw(58) << name
              << " [ Completed 1000 iters ]" << std::endl;
  };

  // 1a. 朴素转置 (Read Coalesced, Write Stride)
  dim3 block32(32, 32);
  dim3 grid32((W + 31) / 32, (H + 31) / 32);
  run_test("1a. Naive Transpose (Read Coalesced, Write Stride)", 1024, 1,
           grid32, block32, [&](dim3 g, dim3 b) {
             kernel_naive_transpose<<<g, b>>>(d_A, d_C, W, H);
           });

  // 1b. 朴素转置 (Read Stride, Write Coalesced)
  run_test("1b. Naive Transpose (Read Stride, Write Coalesced)", 1024, 1,
           grid32, block32, [&](dim3 g, dim3 b) {
             kernel_naive_transpose_swapped<<<g, b>>>(d_A, d_C, W, H);
           });

  // 2a. 冗余 Tiling 原始循环（外 i 内 j -> 单线程写连续 C）
  int tile_w = 4, tile_h = 4;
  dim3 block_no_sm(8, 8);
  dim3 grid_no_sm((W / tile_w + 7) / 8, (H / tile_h + 7) / 8);
  run_test("2a. Redundant Tiling (Outer I, Inner J -> Write Contiguous)", 64,
           16, grid_no_sm, block_no_sm, [&](dim3 g, dim3 b) {
             kernel_tiling_no_shared_transpose<<<g, b>>>(d_A, d_C, W, H, tile_w,
                                                         tile_h);
           });

  // 2b. 冗余 Tiling 调换循环（外 j 内 i -> 单线程读连续 A）
  run_test("2b. Redundant Tiling (Outer J, Inner I -> Read Contiguous)", 64, 16,
           grid_no_sm, block_no_sm, [&](dim3 g, dim3 b) {
             kernel_tiling_no_shared_transpose_swapped<<<g, b>>>(
                 d_A, d_C, W, H, tile_w, tile_h);
           });

  // 3. 基础 Shared Memory 转置 (32x32, 存在 32-way Bank Conflict)
  run_test("3. Shared Mem Transpose (32-way Bank Conflict)", 1024, 1, grid32,
           block32, [&](dim3 g, dim3 b) {
             kernel_shared_memory_transpose<<<g, b>>>(d_A, d_C, W, H);
           });

  // 4. Padding Shared Memory 转置 (32x33, 无 Bank Conflict)
  run_test("4. Padded Shared Mem Transpose (Conflict Free)", 1024, 1, grid32,
           block32, [&](dim3 g, dim3 b) {
             kernel_shared_memory_padding_transpose<<<g, b>>>(d_A, d_C, W, H);
           });

  // 5. 不同 Thread 数量的 Thread Tile + Padding 版本对比 (Block Size 横向对比)
  std::vector<int> block_y_list = {1, 2, 4, 8, 16, 32};
  for (int by : block_y_list) {
    dim3 block(32, by);
    dim3 grid((W + 31) / 32, (H + 31) / 32);
    int threads_num = 32 * by;
    int elem_num = 32 / by;
    std::string name =
        "5. Thread Tile Padded (Block 32x" + std::to_string(by) + ")";

    run_test(name, threads_num, elem_num, grid, block, [&](dim3 g, dim3 b) {
      kernel_shared_memory_tile_padding_transpose<<<g, b>>>(d_A, d_C, W, H);
    });
  }

  // 6. 工业级最标准最高效最终版 (Production-Grade Standard Transpose)
  dim3 block_prod(TILE_DIM, BLOCK_ROWS);
  dim3 grid_prod((W + TILE_DIM - 1) / TILE_DIM, (H + TILE_DIM - 1) / TILE_DIM);
  run_test("6. Production-Grade Standard Transpose (Optimal)", 256, 4,
           grid_prod, block_prod, [&](dim3 g, dim3 b) {
             kernel_transpose_production_optimized<<<g, b>>>(d_A, d_C, W, H);
           });

  // 7. 极致 float4 向量化算子 (float4 Extreme Vectorized Transpose)
  dim3 block_f4(8, 8);
  dim3 grid_f4((W + 31) / 32, (H + 31) / 32);
  run_test("7. Extreme float4 Vectorized Transpose (128-bit)", 64, 16,
           grid_f4, block_f4, [&](dim3 g, dim3 b) {
             kernel_transpose_float4_extreme<<<g, b>>>(d_A, d_C, W, H);
           });

  // 输出完整对比表格
  std::cout << "\n============================================================="
               "====================================================\n";
  std::cout << std::left << std::setw(58) << "Kernel Variant" << std::setw(15)
            << "Threads/Block" << std::setw(15) << "Elem/Thread"
            << std::setw(13) << "Avg Time (ms)" << std::setw(18)
            << "Bandwidth (GB/s)" << std::setw(10) << "Speedup" << "\n";
  std::cout << "---------------------------------------------------------------"
               "--------------------------------------------------\n";

  for (const auto &res : results) {
    std::cout << std::left << std::setw(58) << res.name << std::setw(15)
              << res.threads_per_block << std::setw(15) << res.elem_per_thread
              << std::fixed << std::setprecision(3) << std::setw(13)
              << res.avg_time_ms << std::setw(18) << res.bandwidth_gbps
              << std::setprecision(2) << res.speedup << "x\n";
  }
  std::cout << "==============================================================="
               "==================================================\n\n";

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_C));

  return 0;
}