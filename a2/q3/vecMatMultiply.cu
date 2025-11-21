#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <sys/time.h>

// Evenly split the threads in the x and y directions (max we are allowed is 1024)
#define TX 32
#define TY 32
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

// Get the correct 1D index for a rows x cols matrix, given the (currRow, currCol) coordinates
// Assumes matrix traversed in row-major order (ie, row by row)
int getIdx(int rows, int cols, int currRow, int currCol) {
    // We don't really need "rows" here, but still provided to make signature easy to remember
    return currRow * cols + currCol;
}

// The GPU version of the function to get the index
__device__ int getIdxGpu(int rows, int cols, int currRow, int currCol) {
    return currRow * cols + currCol;
}


// The CPU version of the matrix multiplication kernel
void multiplyCpu(double *matA, double *matB, double *result, int rowsA, int colsA, int rowsB, int colsB) {
    // Since we are doing a A*B, colsA == rowsB is a precondition we assume is true
    // The resulting matrix will be rowsA * colsB

    for (int row = 0; row < rowsA; row++) {
        for (int col = 0; col < colsB; col++) {
            int idx = getIdx(rowsA, colsB, row, col);
            double currSum = 0.0;

            for (int currIdx = 0; currIdx < colsA; currIdx++) {
                currSum += matA[getIdx(rowsA, colsA, row, currIdx)] * matB[getIdx(rowsB, colsB, currIdx, col)];
            }

            result[idx] = currSum;
        }
    }
}

/* The CUDA Add kernel */
__global__ void multiplyGpu(double *matA, double *matB, double *result, int rowsA, int colsA, int rowsB, int colsB) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int idx = getIdxGpu(rowsA, colsB, row, col);
    if (col >= colsB || row >= rowsA) {
        // This thread does not have any specific work to do
        return;
    }

    double currSum = 0.0;
    for (int currIdx = 0; currIdx < colsA; currIdx++) {
        currSum += matA[getIdxGpu(rowsA, colsA, row, currIdx)] * matB[getIdxGpu(rowsB, colsB, currIdx, col)];
    }
    result[idx] = currSum;
}


void compareResult(double *gpuResult, double *cpuResult, int rows, int cols) {
    int allFine = 1;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int idx = getIdx(rows, cols, i, j);
            double diff = gpuResult[i] - cpuResult[i];
            if (fabs(diff) > EPSILON) {
                printf("GPU[%d][%d] = %f, CPU[%d][%d] = %f. %f > %f\n", i, j, gpuResult[idx], i, j, cpuResult[idx], diff, EPSILON);
                allFine = 0;
            }
        }
    }

    if (allFine == 1) {
        printf("All differences were within a reasonable margin! :) \n");
    }
}

void allocateMatrix(double *matrix, int totalRows, int totalCols) {
    for (int row = 0; row < totalRows; row++) {
        for (int col = 0; col < totalCols; col++) {
            int idx = getIdx(totalRows, totalCols, row, col);
            matrix[idx] = ((double) rand()) / RAND_MAX;
        }
    }
}

void printResult(double *gpuResult, double *cpuResult, int rows, int cols) {
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            int idx = getIdx(rows, cols, row, col);
            printf("CPU[%d][%d] = %f, GPU[%d][%d] = %f\n", row, col, cpuResult[idx], row, col, gpuResult[idx]);
        }
    }
}

// The CPU Timer
double cpuSecond() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr, "Usage: %s <rowsA> <colsA> <rowsB> <colsB> <0/1 to check cpu with gpu>\n", argv[0]);
        return 1;
    }

    // Parse the argument for the length of the arrays
    int rowsA = atoi(argv[1]);
    int colsA = atoi(argv[2]);
    int rowsB = atoi(argv[3]);
    int colsB = atoi(argv[4]);
    int checkOutput = atoi(argv[5]);

    if (colsA != rowsB) {
        fprintf(stderr, "A_cols must equal B_rows\n");
        return 1;
    }

    //@@ 1. Allocate in Host Memory
    double *aCpu = (double *)malloc(rowsA * colsA * sizeof(double));
    double *bCpu = (double *)malloc(rowsB * colsB * sizeof(double));
    double *resultCpu = (double *)malloc(rowsA * colsB * sizeof(double));
    double *resultGpuCopy = (double *)malloc(rowsA * colsB * sizeof(double));


    //@@ 2. Allocate in device memory
    double *aGpu, *bGpu, *resultGpu;
    cudaMalloc(&aGpu, rowsA * colsA * sizeof(double));
    cudaMalloc(&bGpu, rowsB * colsB * sizeof(double));
    cudaMalloc(&resultGpu, rowsA * colsB * sizeof(double));

    // // Seed so that the outcome is deterministic
    srand(42);
    //@@ 3. Initialize Host Memory
    allocateMatrix(aCpu, rowsA, colsA);
    allocateMatrix(bCpu, rowsB, colsB);

    // //@@ 4. Copy from host memory to device memory
    double hostDeviceTimeStart = cpuSecond();
    cudaMemcpy(aGpu, aCpu, rowsA * colsA * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(bGpu, bCpu, rowsB * colsB * sizeof(double), cudaMemcpyHostToDevice);
    double hostDeviceTimeElapsed = cpuSecond() - hostDeviceTimeStart;

    // //@ 5. Initialize thread block and thread grid
    dim3 grid((rowsA + TX - 1)/TX, (colsB + TY - 1) / TY, 1);
    dim3 tpb(TX, TY, 1);

    // //@@ 6. Invoke the CUDA Kernel
    double kernelStart = cpuSecond();
    multiplyGpu<<<grid,tpb>>>(aGpu, bGpu, resultGpu, rowsA, colsA, rowsB, colsB);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    double kernelElapsed = cpuSecond() - kernelStart;

    // //@@ 7. Copy results from GPU to CPU
    double deviceHostTimeStart = cpuSecond();
    cudaMemcpy(resultGpuCopy, resultGpu, rowsA * colsB * sizeof(double), cudaMemcpyDeviceToHost);
    double deviceHostTimeElapsed = cpuSecond() - deviceHostTimeStart;

    // //@@ 8. Compare the results with the CPU reference result
    if (checkOutput == 1) {
        multiplyCpu(aCpu, bCpu, resultCpu, rowsA, colsA, rowsB, colsB);
        compareResult(resultGpuCopy, resultCpu, rowsA, colsB);
    }



    // //@@ 9. Free host memory
    free(aCpu);
    free(bCpu);
    free(resultCpu);
    free(resultGpuCopy);

    // //@@ 10. Free device memory
    cudaFree(aGpu);
    cudaFree(bGpu);
    cudaFree(resultGpu);

    printf("host_to_device: %f s\n", hostDeviceTimeElapsed);
    printf("kernel_exec: %f s\n", kernelElapsed);
    printf("device_to_host: %f s\n", deviceHostTimeElapsed);

    return 0;
}
