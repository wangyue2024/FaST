#include "../../include/cuda_timer.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// 1. 行优先合并访存（Coalesced Access Baseline）
__global__ void kernel_coalesced_add(const float *A, const float *B, float *C,
                                     int W, int H) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < W && y < H) {
    int idx = y * W + x;
    C[idx] = A[idx] + B[idx];
  }
}

// 2. 列优先/跨步非合并访存（Col-Major Stride Access）
__global__ void kernel_stride_add(const float *A, const float *B, float *C,
                                  int W, int H) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < W && y < H) {
    int idx = x * H + y; // 跨步读写
    C[idx] = A[idx] + B[idx];
  }
}

// 3. 朴素矩阵转置（Matrix Transpose: Read Coalesced, Write Stride）
__global__ void kernel_naive_transpose(const float *A, float *C, int W, int H) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < W && y < H) {
    C[x * H + y] = A[y * W + x]; // 读连续，写跨步
  }
}

// 4. 基础复杂浮点数计算（Compute-Bound 基础版）
__global__ void kernel_complex_compute(const float *A, const float *B, float *C,
                                       int W, int H) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < W && y < H) {
    int idx = y * W + x;
    float a = A[idx];
    float b = B[idx];
    for (int k = 0; k < 7; ++k) {
      a = sinf(a) * cosf(b) + expf(0.001f * a);
      b = cosf(a) * sinf(b) + sqrtf(fabsf(b) + 1.0f);
    }
    C[idx] = a + b;
  }
}

// 4.5 极限优化版复杂浮点计算 (Hardware SFU + __restrict__ + Unroll + __sincosf)
__global__ void kernel_complex_compute_optimized(const float *__restrict__ A,
                                                 const float *__restrict__ B,
                                                 float *__restrict__ C, int W,
                                                 int H) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= W || y >= H)
    return;

  int idx = y * W + x;
  float a = A[idx];
  float b = B[idx];

  // 0.001 * log2(e)，编译期常数折叠
  constexpr float COEFF = 0.001f * 1.4426950408889634f;

#pragma unroll
  for (int k = 0; k < 20; ++k) {
    float sa, ca, sb, cb;
    // 1. 硬件级同时提取正余弦
    __sincosf(a, &sa, &ca);
    __sincosf(b, &sb, &cb);

    // 2. 硬件直连快速 2 进制指数与开方
    a = sa * cb + exp2f(a * COEFF);
    b = ca * sb + __fsqrt_rn(fabsf(b) + 1.0f);
  }

  C[idx] = a + b;
}

// 5. 彻底完全随机访存（Random Access - L2 Cache 击穿 & 8x 惩罚极限测试）
__global__ void kernel_random_access(const float *A, const float *B, float *C,
                                     const int *rand_idx, int W, int H) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < W && y < H) {
    int tid = y * W + x;
    int r_idx = rand_idx[tid]; // 完全随机离散索引
    C[tid] = A[r_idx] + B[r_idx];
  }
}

struct TestResult {
  std::string name;
  float avg_time_ms;
  double bandwidth_gbps;
  float slowdown;
};

int main() {
  const int W = 1 << 12; // 4096
  const int H = 1 << 12; // 4096
  const int N = W * H;
  const size_t bytes = N * sizeof(float);
  const size_t idx_bytes = N * sizeof(int);

  std::cout << "==============================================================="
               "==========================\n";
  std::cout << "          CUDA Multi-Kernel Benchmark (Including SFU Optimized "
               "Kernel)                  \n";
  std::cout << "Matrix Size: " << W << " x " << H << " (" << N << " elements, "
            << (bytes / (1024.0 * 1024.0)) << " MB)\n";
  std::cout << "==============================================================="
               "==========================\n\n";

  float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
  int *d_rand_idx = nullptr;
  CUDA_CHECK(cudaMalloc(&d_A, bytes));
  CUDA_CHECK(cudaMalloc(&d_B, bytes));
  CUDA_CHECK(cudaMalloc(&d_C, bytes));
  CUDA_CHECK(cudaMalloc(&d_rand_idx, idx_bytes));

  CUDA_CHECK(cudaMemset(d_A, 1, bytes));
  CUDA_CHECK(cudaMemset(d_B, 2, bytes));
  CUDA_CHECK(cudaMemset(d_C, 0, bytes));

  std::cout << "Generating Random Permutation Index Array on Host..."
            << std::flush;
  std::vector<int> h_rand_idx(N);
  std::iota(h_rand_idx.begin(), h_rand_idx.end(), 0);
  std::mt19937 g(1337);
  std::shuffle(h_rand_idx.begin(), h_rand_idx.end(), g);
  CUDA_CHECK(cudaMemcpy(d_rand_idx, h_rand_idx.data(), idx_bytes,
                        cudaMemcpyHostToDevice));
  std::cout << " [ Done ]\n\n";

  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((W + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (H + threadsPerBlock.y - 1) / threadsPerBlock.y);

  const int warmup_iters = 20;
  const int num_iters = 200;
  CudaTimer timer;
  std::vector<TestResult> results;

  // 1. 合并访存 Baseline
  for (int i = 0; i < warmup_iters; ++i)
    kernel_coalesced_add<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, W,
                                                             H);
  CUDA_CHECK(cudaDeviceSynchronize());

  timer.start();
  for (int i = 0; i < num_iters; ++i)
    kernel_coalesced_add<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, W,
                                                             H);
  float t1 = timer.stop() / num_iters;
  double bw1 = (3.0 * bytes) / (t1 * 1e6);
  results.push_back({"1. Row-Major Coalesced Add (Baseline)", t1, bw1, 1.0f});

  // 2. 非合并访存
  for (int i = 0; i < warmup_iters; ++i)
    kernel_stride_add<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, W, H);
  CUDA_CHECK(cudaDeviceSynchronize());

  timer.start();
  for (int i = 0; i < num_iters; ++i)
    kernel_stride_add<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, W, H);
  float t2 = timer.stop() / num_iters;
  double bw2 = (3.0 * bytes) / (t2 * 1e6);
  results.push_back({"2. Col-Major Non-Coalesced Add", t2, bw2, t2 / t1});

  // 3. 朴素转置
  for (int i = 0; i < warmup_iters; ++i)
    kernel_naive_transpose<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C, W, H);
  CUDA_CHECK(cudaDeviceSynchronize());

  timer.start();
  for (int i = 0; i < num_iters; ++i)
    kernel_naive_transpose<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C, W, H);
  float t3 = timer.stop() / num_iters;
  double bw3 = (2.0 * bytes) / (t3 * 1e6);
  results.push_back({"3. Naive Matrix Transpose", t3, bw3, t3 / t1});

  // 4. 复杂浮点基础版
  for (int i = 0; i < warmup_iters; ++i)
    kernel_complex_compute<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, W,
                                                               H);
  CUDA_CHECK(cudaDeviceSynchronize());

  timer.start();
  for (int i = 0; i < num_iters; ++i)
    kernel_complex_compute<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, W,
                                                               H);
  float t4 = timer.stop() / num_iters;
  double bw4 = (3.0 * bytes) / (t4 * 1e6);
  results.push_back(
      {"4. Complex Floating Compute (Standard)", t4, bw4, t4 / t1});

  // 4.5 极限优化版复杂浮点计算 (SFU + sincosf + unroll)
  for (int i = 0; i < warmup_iters; ++i)
    kernel_complex_compute_optimized<<<blocksPerGrid, threadsPerBlock>>>(
        d_A, d_B, d_C, W, H);
  CUDA_CHECK(cudaDeviceSynchronize());

  timer.start();
  for (int i = 0; i < num_iters; ++i)
    kernel_complex_compute_optimized<<<blocksPerGrid, threadsPerBlock>>>(
        d_A, d_B, d_C, W, H);
  float t4_opt = timer.stop() / num_iters;
  double bw4_opt = (3.0 * bytes) / (t4_opt * 1e6);
  results.push_back({"4.5 Complex Compute (SFU + sincos + unroll)", t4_opt,
                     bw4_opt, t4_opt / t1});

  // 5. 纯随机访存
  for (int i = 0; i < warmup_iters; ++i)
    kernel_random_access<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C,
                                                             d_rand_idx, W, H);
  CUDA_CHECK(cudaDeviceSynchronize());

  timer.start();
  for (int i = 0; i < num_iters; ++i)
    kernel_random_access<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C,
                                                             d_rand_idx, W, H);
  float t5 = timer.stop() / num_iters;
  double bw5 = (4.0 * bytes) / (t5 * 1e6);
  results.push_back(
      {"5. Pure Random Access (L2 Cache Destroyed)", t5, bw5, t5 / t1});

  // 打印表格
  std::cout << std::fixed << std::setprecision(3);
  std::cout << std::left << std::setw(50) << "Kernel Execution Mode"
            << std::setw(15) << "Avg Time (ms)" << std::setw(18)
            << "Bandwidth (GB/s)" << std::setw(12) << "Slowdown" << "\n";
  std::cout << "---------------------------------------------------------------"
               "------------------------------------------\n";

  for (const auto &res : results) {
    std::cout << std::left << std::setw(50) << res.name << std::setw(15)
              << res.avg_time_ms << std::setw(18) << res.bandwidth_gbps
              << res.slowdown << "x\n";
  }
  std::cout << "==============================================================="
               "=================-------------------------\n";

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
  CUDA_CHECK(cudaFree(d_rand_idx));

  return 0;
}