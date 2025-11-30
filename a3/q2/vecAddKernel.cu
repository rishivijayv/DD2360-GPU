#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <sys/time.h>

// We will have the maximum number of threads per block allowed
#define TPB 1024
// The margin of error that we will accept for a difference between CPU sums and GPU sums
#define EPSILON 0.001
// The number of streams to use
#define STREAMS 4

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
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <length of array to sum> <segment length> <0/1 to check cpu with gpu> <0/1 to use normal vector addition>\n", argv[0]);
        return 1;
    }

    // Parse the argument for the length of the arrays
    int n = atoi(argv[1]);
    int segmentLength = atoi(argv[2]);
    int checkOutput = atoi(argv[3]);
    int useNormalAdd = atoi(argv[4]);

    //@@ 1. Allocate in Host Memory. Using pinned memory so that async transfer can be performed
    double *aCpu, *bCpu, *cCpu, *cGpuCopy, *cGpuStreamCopy;
    cudaHostAlloc((void **) &aCpu, n * sizeof(double), cudaHostAllocDefault);
    cudaHostAlloc((void **) &bCpu, n * sizeof(double), cudaHostAllocDefault);
    cudaHostAlloc((void **) &cCpu, n * sizeof(double), cudaHostAllocDefault);
    cudaHostAlloc((void **) &cGpuCopy, n * sizeof(double), cudaHostAllocDefault);
    cudaHostAlloc((void **) &cGpuStreamCopy, n * sizeof(double), cudaHostAllocDefault);

    //@@ 2. Allocate in device memory
    double *aGpu, *bGpu, *cGpu, *aGpuStream, *bGpuStream, *cGpuStream;
    cudaMalloc(&aGpu, n * sizeof(double));
    cudaMalloc(&bGpu, n * sizeof(double));
    cudaMalloc(&cGpu, n * sizeof(double));
    cudaMalloc(&aGpuStream, n * sizeof(double));
    cudaMalloc(&bGpuStream, n * sizeof(double));
    cudaMalloc(&cGpuStream, n * sizeof(double));

    // Seed so that the outcome is deterministic
    srand(42);
    //@@ 3. Initialize Host Memory
    for (int i = 0; i < n; i++) {
        aCpu[i] = ((double) rand()) / RAND_MAX;
        bCpu[i] = ((double) rand()) / RAND_MAX;
    }

    // ============== VECTOR ADDITION THE NORMAL WAY ===================
    double normalElapsed;
    if (useNormalAdd == 1) {
        double normalStart = cpuSecond();
        //@@ 4. Copy from host memory to device memory
        cudaMemcpy(aGpu, aCpu, n * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(bGpu, bCpu, n * sizeof(double), cudaMemcpyHostToDevice);

        //@ 5. Initialize thread block and thread grid
        dim3 grid((n + TPB - 1)/TPB, 1, 1);
        dim3 tpb(TPB, 1, 1);

        //@@ 6. Invoke the CUDA Kernel
        addGpu<<<grid,tpb>>>(aGpu, bGpu, cGpu, n);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        //@@ 7. Copy results from GPU to CPU
        cudaMemcpy(cGpuCopy, cGpu, n * sizeof(double), cudaMemcpyDeviceToHost);
        normalElapsed = cpuSecond() - normalStart;
    }



    // ================= VECTOR ADDITION THE STREAM WAY ==================
    cudaStream_t stream[STREAMS];
    for (int i = 0; i < STREAMS; i++) {
        cudaStreamCreate(&stream[i]);
    }

    double streamStart = cpuSecond();
    int numSegments = (n + segmentLength - 1) / segmentLength;

    // Segments will be allocated in round-robin fashion.
    // This means segment 0 to stream 0, segment 1 to stream 1, and so on.
    // That is, segment i goes to stream i % 4
    for (int i = 0; i < numSegments; i++) {
        int streamId = i % STREAMS;
        int segmentStart = i * segmentLength;
        int totalElements = segmentLength;
        if (segmentStart + totalElements > n) {
            // If this is the last segment, there could be fewer elements to work on for the stream
            // in the case that n is not completely divisible by segment length
            totalElements = n - segmentStart;
        }

        // Async copy from device to host
        cudaMemcpyAsync(&aGpuStream[segmentStart], &aCpu[segmentStart], totalElements * sizeof(double), cudaMemcpyHostToDevice, stream[streamId]);
        cudaMemcpyAsync(&bGpuStream[segmentStart], &bCpu[segmentStart], totalElements * sizeof(double), cudaMemcpyHostToDevice, stream[streamId]);

        // Execute the kernel 
        // Each "segment" is now a small vector addition, so determine number of blocks based on the number of elements in the current segment
        dim3 gridStream((totalElements + TPB - 1) / TPB, 1 , 1);
        dim3 tpbStream(TPB, 1, 1);
        addGpu<<<gridStream, tpbStream, 0, stream[streamId]>>>(&aGpuStream[segmentStart], &bGpuStream[segmentStart], &cGpuStream[segmentStart], totalElements);

        // Async copt from host to device
        cudaMemcpyAsync(&cGpuStreamCopy[segmentStart], &cGpuStream[segmentStart], totalElements * sizeof(double), cudaMemcpyDeviceToHost, stream[streamId]);
    }
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    double streamElapsed = cpuSecond() - streamStart;

    for (int i = 0; i < STREAMS; i++) {
        cudaStreamDestroy(stream[i]);
    }

    //@@ 8. Compare the results with the CPU reference result
    if (checkOutput == 1) {
        addCpu(aCpu, bCpu, cCpu, n);
        if (useNormalAdd == 1) {
            compareResult(cGpuCopy, cCpu, n);
        }
        compareResult(cGpuStreamCopy, cCpu, n);
    }


    //@@ 9. Free host memory
    cudaFreeHost(aCpu);
    cudaFreeHost(bCpu);
    cudaFreeHost(cCpu);
    cudaFreeHost(cGpuCopy);
    cudaFreeHost(cGpuStreamCopy);

    //@@ 10. Free device memory
    cudaFree(aGpu);
    cudaFree(bGpu);
    cudaFree(cGpu);
    cudaFree(aGpuStream);
    cudaFree(bGpuStream);
    cudaFree(cGpuStream);
    
    if (useNormalAdd == 1) {
        printf("normal: %f ms\n", normalElapsed * 1000);
    }
    printf("streams: %f ms\n", streamElapsed * 1000);

    return 0;
}
