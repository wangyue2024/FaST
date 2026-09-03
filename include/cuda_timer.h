#pragma once
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << " - "    \
                << cudaGetErrorString(err) << std::endl;                       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

class CudaTimer {
private:
  cudaEvent_t start_event, stop_event;

public:
  CudaTimer() {
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));
  }

  ~CudaTimer() {
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));
  }

  void start() { CUDA_CHECK(cudaEventRecord(start_event)); }

  float stop() {
    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_event, stop_event));
    return ms;
  }
};
