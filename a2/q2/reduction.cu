#include <random>
#include <chrono>
#include <iostream>
#include <string>

#define CHECK(call) do {                                 \
    cudaError_t err = (call);                            \
    if (err != cudaSuccess) {                            \
        std::fprintf(stderr, "CUDA error: %s (%s:%d)\n", \
                     cudaGetErrorString(err), __FILE__, __LINE__); \
        std::exit(1);                                    \
    }                                                    \
} while (0)



__global__ void reduction_kernel_naive(float *d_a, float *d_res, int input_length) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    	if (idx >= input_length) return;

	atomicAdd(d_res, d_a[idx]);
	return;
}


__global__ void reduction_kernel_tiling(float *d_a, float *d_res, int input_length) {
	__shared__ float l_res;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    	if (idx >= input_length) return;

    	if (idx >= input_length) return;
	if (threadIdx.x == 0) l_res = 0.0;
	__syncthreads();

	atomicAdd(&l_res, d_a[idx]);
	__syncthreads();

	if (threadIdx.x == 0) atomicAdd(d_res, l_res);
	return;
}

__global__ void reduction_kernel_tiling_mem_adj(float *d_a, float *d_res, int input_length, int elements_per_thread) {
	__shared__ float l_res;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    	if (idx >= input_length) return;
	if (idx + elements_per_thread >= input_length) elements_per_thread = input_length - idx;
	if (threadIdx.x == 0) l_res = 0.0;

	float t_res = 0.0;
	
	for (int i = 0; i < elements_per_thread; ++i) t_res += d_a[idx + i];
	__syncthreads();

	atomicAdd(&l_res, t_res);
	__syncthreads();

	if (threadIdx.x == 0) atomicAdd(d_res, l_res);
	return;
}


static void initialise(float* a, const int input_length) {
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    //std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::mt19937 gen; // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<float> dis(0.0, 1.0);

    for (int i = 0; i < input_length; ++i) a[i] = dis(gen);
}


static float compute_reference_solution(const float *a, const int input_length) {
    double reference_solution = 0.0;
    for (int i = 0; i < input_length; ++i) {
        reference_solution += static_cast<double>(a[i]);
    }

    return static_cast<float>(reference_solution);
}


int main(int argc, char **argv) {
    int input_length = 21e8; // close to max int size
    //int input_length = pow(32, 6); // close to max int size

    if (argc == 2) {
        input_length = std::stoi(argv[1]);
    }

    printf("\nInitialise:\n\tInput length: %d \n", input_length);
    //@@ Insert code below to initialize the input array with random values on CPU

    auto init_start = std::chrono::system_clock::now();
    auto *a = static_cast<float *>(malloc(input_length * sizeof(float)));
    initialise(a, input_length);
    auto init_end = std::chrono::system_clock::now();
    auto init_diff = std::chrono::duration_cast<std::chrono::milliseconds>(init_end - init_start);

    printf("\t...took: %ld ms\n\n", init_diff.count());

    //@@ Insert code below to create reference result in CPU and add a timer

    printf("\nCompute reference solution:\n");
    auto ref_start = std::chrono::system_clock::now();
    float ref_res = compute_reference_solution(a, input_length);
    auto ref_end = std::chrono::system_clock::now();
    auto ref_diff = std::chrono::duration_cast<std::chrono::milliseconds>(ref_end - ref_start);
    printf("\tReference solution is: %f\n", ref_res/static_cast<float>(input_length));
    printf("\t...took: %ld ms\n\n", ref_diff.count());

    //@@ Insert code to copy data from CPU to the GPU

    printf("\nCopy data to device:\n");
    auto cpy_start = std::chrono::system_clock::now();

    float *d_a = nullptr;
    cudaMalloc(&d_a, input_length * sizeof(float));
    cudaMemcpy(d_a, a, input_length * sizeof(float), cudaMemcpyHostToDevice);

    float *d_res = nullptr;
    cudaMalloc(&d_res, 1 * sizeof(float));
    cudaMemset(d_res, 0, 1 * sizeof(float));

    auto cpy_end = std::chrono::system_clock::now();
    auto cpy_diff = std::chrono::duration_cast<std::chrono::milliseconds>(cpy_end - cpy_start);
    printf("\t...took: %ld ms\n\n", cpy_diff.count());

    //@@ Initialize the grid and block dimensions here

    if (input_length == 0) return -1;
    int TPB = 1024;
    int NBlocks = 1 + ((input_length - 1) / TPB); // assumes input_length != 0; computes ceil(q) with q = x / y (int division).


    //@@ Launch the GPU Kernel here and add a timer and copy the GPU memory back to the CPU here

    //Naive:

    printf("\nLaunching naive kernel:\n\tThreads per Block: %d \n\tNumber of Blocks: %d \n\tTotal number of Threads: %d\n", TPB, NBlocks, TPB * NBlocks);

    auto naive_start = std::chrono::system_clock::now();

    reduction_kernel_naive<<<NBlocks,TPB>>>(d_a, d_res, input_length);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());


    float naive_res = 0.0;
    cudaMemcpy(&naive_res, d_res, 1 * sizeof(float), cudaMemcpyDeviceToHost); // synchronises, no barrier needed afaik.

    auto naive_end = std::chrono::system_clock::now();
    auto naive_diff = std::chrono::duration_cast<std::chrono::milliseconds>(naive_end - naive_start);
    printf("\tNaive solution is: %f\n", naive_res/static_cast<float>(input_length));
    printf("\t...took: %ld ms\n\n", naive_diff.count());

    // Tiling:
    
    cudaMemset(d_res, 0, 1 * sizeof(float)); // Reset Memory, don't measure to not double count (already incldued in initialisation).
    printf("\nLaunching tiling kernel:\n\tThreads per Block: %d \n\tNumber of Blocks: %d \n\tTotal number of Threads: %d\n", TPB, NBlocks, TPB * NBlocks);

    auto tiling_start = std::chrono::system_clock::now();
    reduction_kernel_tiling<<<NBlocks,TPB>>>(d_a, d_res, input_length);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    float tiling_res = 0.0;
    cudaMemcpy(&tiling_res, d_res, 1 * sizeof(float), cudaMemcpyDeviceToHost); // synchronises, no barrier needed afaik.

    auto tiling_end = std::chrono::system_clock::now();
    auto tiling_diff = std::chrono::duration_cast<std::chrono::milliseconds>(tiling_end - tiling_start);
    printf("\tTiling solution is: %f\n", tiling_res/static_cast<float>(input_length));
    printf("\t...took: %ld ms\n\n", tiling_diff.count());

    // Tiling with cache optimisations:
    //for (int i = 0; i <= 20; ++i){

	    //int elements_per_thread = static_cast<int>(pow(2, i));
	    int elements_per_thread = static_cast<int>(pow(2, 13));
	    int NBlocks_adjusted = NBlocks / elements_per_thread;

	    cudaMemset(d_res, 0, 1 * sizeof(float)); // Reset Memory, don't measure to not double count (already incldued in initialisation).
	    printf("\nLaunching tiling kernel with memory opt:\n\tThreads per Block: %d \n\tNumber of Blocks: %d \n\tTotal number of Threads: %d\n\tElements per Thread: %d\n", TPB, NBlocks_adjusted, TPB * NBlocks_adjusted, elements_per_thread);

	    auto tiling2_start = std::chrono::system_clock::now();
	    reduction_kernel_tiling_mem_adj<<<NBlocks_adjusted,TPB>>>(d_a, d_res, input_length, elements_per_thread);

	    CHECK(cudaGetLastError());
	    CHECK(cudaDeviceSynchronize());

	    float tiling2_res = 0.0;
	    cudaMemcpy(&tiling2_res, d_res, 1 * sizeof(float), cudaMemcpyDeviceToHost); // synchronises, no barrier needed afaik.

	    auto tiling2_end = std::chrono::system_clock::now();
	    auto tiling2_diff = std::chrono::duration_cast<std::chrono::milliseconds>(tiling2_end - tiling2_start);
	    printf("\tTiling solution is: %f\n", tiling2_res/static_cast<float>(input_length));
	    printf("\t...took: %ld ms\n\n", tiling2_diff.count());
    //}



    //@@ Free memory here
    free(a);
    cudaFree(d_a);
    cudaFree(d_res);

    return 0;
}
