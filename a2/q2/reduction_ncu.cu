#include <chrono>
#include <iostream>
#include <random>
#include <string>

#define CHECK(call)                                                            \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA error: %s (%s:%d)\n",                         \
                   cudaGetErrorString(err), __FILE__, __LINE__);               \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

__global__ void reduction_kernel_naive(float *d_a, float *d_res,
                                       int input_length) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= input_length)
    return;

  atomicAdd(d_res, d_a[idx]);
  return;
}

__global__ void reduction_kernel_tiling(float *d_a, float *d_res,
                                        int input_length) {
  __shared__ float l_res;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= input_length)
    return;

  if (idx >= input_length)
    return;
  if (threadIdx.x == 0)
    l_res = 0.0;
  __syncthreads();

  atomicAdd(&l_res, d_a[idx]);
  __syncthreads();

  if (threadIdx.x == 0)
    atomicAdd(d_res, l_res);
  return;
}

__global__ void reduction_kernel_tiling_mem_adj(float *d_a, float *d_res,
                                                int input_length,
                                                int elements_per_thread) {
  __shared__ float l_res;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= input_length)
    return;
  if (idx + elements_per_thread >= input_length)
    elements_per_thread = input_length - idx;
  if (threadIdx.x == 0)
    l_res = 0.0;

  float t_res = 0.0;

  for (int i = 0; i < elements_per_thread; ++i)
    t_res += d_a[idx + i];
  __syncthreads();

  atomicAdd(&l_res, t_res);
  __syncthreads();

  if (threadIdx.x == 0)
    atomicAdd(d_res, l_res);
  return;
}

static void initialise(float *a, const int input_length) {
  std::random_device
      rd; // Will be used to obtain a seed for the random number engine
  // std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with
  // rd()
  std::mt19937 gen; // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<float> dis(0.0, 1.0);

  for (int i = 0; i < input_length; ++i)
    a[i] = dis(gen);
}

static float compute_reference_solution(const float *a,
                                        const int input_length) {
  double reference_solution = 0.0;
  for (int i = 0; i < input_length; ++i) {
    reference_solution += static_cast<double>(a[i]);
  }

  return static_cast<float>(reference_solution);
}

void benchmark(float *a, int input_length) {
  auto cpy_start = std::chrono::system_clock::now();
  float *d_a = nullptr;
  cudaMalloc(&d_a, input_length * sizeof(float));
  cudaMemcpy(d_a, a, input_length * sizeof(float), cudaMemcpyHostToDevice);

  float *d_res = nullptr;
  cudaMalloc(&d_res, 1 * sizeof(float));
  cudaMemset(d_res, 0, 1 * sizeof(float));

  auto cpy_end = std::chrono::system_clock::now();
  auto cpy_diff =
      std::chrono::duration_cast<std::chrono::nanoseconds>(cpy_end - cpy_start);
  printf("\t...took: %ld ns\n\n", cpy_diff.count());

  int TPB = 1024;
  int NBlocks = 1 + ((input_length - 1) / TPB);
  int elements_per_thread = static_cast<int>(pow(2, 10));
  int NBlocks_adjusted = NBlocks / elements_per_thread;

  auto tiling2_start = std::chrono::system_clock::now();
  reduction_kernel_tiling_mem_adj<<<NBlocks_adjusted, TPB>>>(
      d_a, d_res, input_length, elements_per_thread);

  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  float tiling2_res = 0.0;
  cudaMemcpy(&tiling2_res, d_res, 1 * sizeof(float),
             cudaMemcpyDeviceToHost); // synchronises, no barrier needed afaik.

  auto tiling2_end = std::chrono::system_clock::now();
  auto tiling2_diff = std::chrono::duration_cast<std::chrono::nanoseconds>(
      tiling2_end - tiling2_start);
  printf("\t...took: %ld ns\n\n", tiling2_diff.count());

  //@@ Free memory here
  cudaFree(d_a);
  cudaFree(d_res);
}

void init_benchmark(float **a, int input_length) {
  auto init_start = std::chrono::system_clock::now();
  *a = static_cast<float *>(malloc(input_length * sizeof(float)));
  initialise(*a, input_length);
  auto init_end = std::chrono::system_clock::now();
  auto init_diff = std::chrono::duration_cast<std::chrono::milliseconds>(
      init_end - init_start);

  printf("\t...took: %ld ms\n\n", init_diff.count());
}

int main(int argc, char **argv) {
  int input_length = static_cast<int>(pow(2, 30));
  printf("\nInitialise:\n\tInput length: 2**%d = %d\n", 30, input_length);
  float *a = nullptr;
  init_benchmark(&a, input_length);
  benchmark(a, input_length);
  free(a);
  return 0;
}
