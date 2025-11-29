#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <sys/time.h>

// We will have the maximum number of threads per block allowed
#define TPB 1024
// The margin of error that we will accept for a difference between CPU sums and GPU sums
#define EPSILON 0.001

#define CHECK(call) do {                                 \
    cudaError_t err = (call);                            \
    if (err != cudaSuccess) {                            \
        std::fprintf(stderr, "CUDA error: %s (%s:%d)\n", \
                     cudaGetErrorString(err), __FILE__, __LINE__); \
        std::exit(1);                                    \
    }                                                    \
} while (0)

/* The CUDA Add kernel */
__global__ void addGpu(double *a, double *b, double *c, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        // This thread does not have any specific work to do
        return;
    }
    c[idx] = a[idx] + b[idx];
}


// The CPU version of the "add" kernel
void addCpu(double *a, double *b, double *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

/**
 Debugging function to print results in a more "beautiful" manner
*/
void printResults(double *a, double *b, double *c, int n, const char *heading) {
    printf("%s\n", heading);
    printf("Index\tArray A\tArray B\tSum\n");
    printf("------\t-------\t-------\t-------\n");
    for (int i = 0; i < n; i++) {
        printf("%d\t%f\t%f\t%f\n", i, a[i], b[i], c[i]);
    }
}

void compareResult(double *gpuResult, double *cpuResult, int n) {
    int allFine = 1;


    for (int i = 0; i < n; i++) {
        // printf("GPU[%d] = %f, CPU[%d] = %f\n", i, gpuResult[i], i, cpuResult[i]);
        double diff = gpuResult[i] - cpuResult[i];
        if (fabs(diff) > EPSILON) {
            printf("GPU[%d] = %f, CPU[%d] = %f. %f > %f\n", i, gpuResult[i], i, cpuResult[i], diff, EPSILON);
            allFine = 0;
        }
    }

    if (allFine == 1) {
        printf("All differences were within a reasonable margin! :) \n");
    }
}

// The CPU Timer
double cpuSecond() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <length of array to sum> <0/1 to check cpu with gpu>", argv[0]);
        return 1;
    }

    // Parse the argument for the length of the arrays
    int n = atoi(argv[1]);
    int checkOutput = atoi(argv[2]);

    //@@ 1. Allocate in Host Memory
    double *aCpu = (double *)malloc(n * sizeof(double));
    double *bCpu = (double *)malloc(n * sizeof(double));
    double *cCpu = (double *)malloc(n * sizeof(double));
    double *cGpuCopy = (double *)malloc(n * sizeof(double));

    //@@ 2. Allocate in device memory
    double *aGpu, *bGpu, *cGpu;
    cudaMalloc(&aGpu, n * sizeof(double));
    cudaMalloc(&bGpu, n * sizeof(double));
    cudaMalloc(&cGpu, n * sizeof(double));

    // Seed so that the outcome is deterministic
    srand(42);
    //@@ 3. Initialize Host Memory
    for (int i = 0; i < n; i++) {
        aCpu[i] = ((double) rand()) / RAND_MAX;
        bCpu[i] = ((double) rand()) / RAND_MAX;
    }

    //@@ 4. Copy from host memory to device memory
    double hostDeviceTimeStart = cpuSecond();
    cudaMemcpy(aGpu, aCpu, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(bGpu, bCpu, n * sizeof(double), cudaMemcpyHostToDevice);
    double hostDeviceTimeElapsed = cpuSecond() - hostDeviceTimeStart;

    //@ 5. Initialize thread block and thread grid
    dim3 grid((n + TPB - 1)/TPB, 1, 1);
    dim3 tpb(TPB, 1, 1);

    //@@ 6. Invoke the CUDA Kernel
    double kernelStart = cpuSecond();
    addGpu<<<grid,tpb>>>(aGpu, bGpu, cGpu, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    double kernelElapsed = cpuSecond() - kernelStart;

    //@@ 7. Copy results from GPU to CPU
    double deviceHostTimeStart = cpuSecond();
    cudaMemcpy(cGpuCopy, cGpu, n * sizeof(double), cudaMemcpyDeviceToHost);
    double deviceHostTimeElapsed = cpuSecond() - deviceHostTimeStart;

    //@@ 8. Compare the results with the CPU reference result
    if (checkOutput == 1) {
        addCpu(aCpu, bCpu, cCpu, n);
        compareResult(cGpuCopy, cCpu, n);
    }


    //@@ 9. Free host memory
    free(aCpu);
    free(bCpu);
    free(cCpu);
    free(cGpuCopy);

    //@@ 10. Free device memory
    cudaFree(aGpu);
    cudaFree(bGpu);
    cudaFree(cGpu);

    printf("host_to_device: %f s\n", hostDeviceTimeElapsed);
    printf("kernel_exec: %f s\n", kernelElapsed);
    printf("device_to_host: %f s\n", deviceHostTimeElapsed);

    return 0;
}
