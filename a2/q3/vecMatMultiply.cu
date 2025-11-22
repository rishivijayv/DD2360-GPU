#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <sys/time.h>

// Evenly split the threads in the x and y directions (max we are allowed is 1024)
#define TX 32
#define TY 32
// The margin of error that we will accept for a difference between CPU sums and GPU sums
#define EPSILON 0.001

// The tile sizes to sweep through
#define TILE_SIZE_1 8
#define TILE_SIZE_2 16
#define TILE_SIZE_3 32


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
void gemm_cpu(double *matA, double *matB, double *result, int rowsA, int colsA, int rowsB, int colsB) {
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
__global__ void gemm(double *matA, double *matB, double *result, int rowsA, int colsA, int rowsB, int colsB) {
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


/* The TILED version of the CUDA Kernel */
__global__ void tiled_gemm(double *matA, double *matB, double *result, int rowsA, int colsA, int rowsB, int colsB, int tileX, int tileY) {
    // Assume that colsA = rowsB
    // We will divide A into tiles of sizes (tileX x tileY), and B into tiles of sizes (tileY, tileX).
    // Thus, in each sweep, we will move the block for A forward by **tileY** columns and the block for B forward by **tileY** rows.
    // So, each block will compute (tileX x tileX) elements of C. Ideally to make sure that all threads in a block do some 
    // computation (unless we're at boundary conditions), we would want tileX = tileY.

    // Global rows and cols in the matrix that this thread could be responsible for
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Some constants for this thread
    const int tileIncrement = tileY;
    const int localRow = threadIdx.y, localCol = threadIdx.x;

    // Dynamic shared array 
    extern __shared__ double both_matrix_tiles[];

    // Pointer math to split the shared array in to chunks
    double *A = (double *) both_matrix_tiles;
    double *B = (double *) (both_matrix_tiles + (tileX * tileY));

    // Will be used to accumulate the result
    double accResult = 0.0;
    int numTiles = (colsA + tileIncrement - 1) / tileIncrement;
    int performedComputation = 0;

    // Sweep through the tiles
    for (int i = 0; i < numTiles; i++) {
        // Conditionally load elements of A if they are in range. If they are not in range, load 0.0
        if (row < rowsA && ((i * tileIncrement) + localCol) < colsA) {
            A[getIdxGpu(tileX, tileY, localRow, localCol)] = matA[getIdxGpu(rowsA, colsA, row, (i * tileIncrement) + localCol)];
        } else {
            A[getIdxGpu(tileX, tileY, localRow, localCol)] = 0.0;
        }

        // Do the same when loading elements of B
        if (((i * tileIncrement) + localRow) < rowsB && col < colsB) {
            B[getIdxGpu(tileY, tileX, localRow, localCol)] = matB[getIdxGpu(rowsB, colsB, (i * tileIncrement) + localRow, col)];
        } else {
            B[getIdxGpu(tileY, tileX, localRow, localCol)] = 0.0;
        }

        // Wait until all threads have finished loading elements
        __syncthreads();

        // Only do the computation if the thread is assigned to compute for the (tileX x tileX) block of C
        if (localRow < tileX && localCol < tileX) {
            // This thread performed computation, so will be setting a value for C
            performedComputation = 1;

            // Get the (localRow, localCol) answer by multiplying A[localRow][:] x B[:][localCol]
            for (int j = 0; j < tileY; j++) {
                accResult += A[getIdxGpu(tileX, tileY, localRow, j)] * B[getIdxGpu(tileY, tileX, j, localCol)];
            }
        }

        // Wait for all threads to finish computation before sweeping the next set of tiles
        __syncthreads();
    }

    // Conditionally set the result if 
    if (row < rowsA && col < colsB && performedComputation == 1) {
        result[getIdxGpu(rowsA, colsB, row, col)] = accResult;
    }


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
    double ctr = 0;
    for (int row = 0; row < totalRows; row++) {
        for (int col = 0; col < totalCols; col++) {
            int idx = getIdx(totalRows, totalCols, row, col);
            matrix[idx] = ((double) rand()) / RAND_MAX;
            ctr++;
        }
    }
}

void printMatrix(double *matrix, int rows, int cols) {
    int rowLimit = rows > 5 ? 5 : rows;
    int colLimit = cols > 5 ? 5 : cols;
    for (int row = 0; row < rowLimit; row++) {
        for (int col = 0; col < colLimit; col++) {
            int idx = getIdx(rows, cols, row, col);
            printf("%f  ", matrix[idx]);
        }
        printf("\n");
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
        fprintf(stderr, "Usage: %s <rowsA> <colsA> <rowsB> <colsB> <0=print_timing_for_graphing,1=print_user_friendly_output>\n", argv[0]);
        return 1;
    }

    // Parse the argument for the length of the arrays
    int rowsA = atoi(argv[1]);
    int colsA = atoi(argv[2]);
    int rowsB = atoi(argv[3]);
    int colsB = atoi(argv[4]);
    int printPretty = atoi(argv[5]);

    if (colsA != rowsB) {
        fprintf(stderr, "A_cols must equal B_rows\n");
        return 1;
    }

    //@@ 1. Allocate in Host Memory
    double *aCpu = (double *)malloc(rowsA * colsA * sizeof(double));
    double *bCpu = (double *)malloc(rowsB * colsB * sizeof(double));
    double *resultCpu = (double *)malloc(rowsA * colsB * sizeof(double));
    double *resultGpuCopy = (double *)malloc(rowsA * colsB * sizeof(double));
    double *resultGpuCopyTile1 = (double *)malloc(rowsA * colsB * sizeof(double));
    double *resultGpuCopyTile2 = (double *)malloc(rowsA * colsB * sizeof(double));
    double *resultGpuCopyTile3 = (double *)malloc(rowsA * colsB * sizeof(double));



    //@@ 2. Allocate in device memory
    double *aGpu, *bGpu, *resultGpu, *resultGpuTile1, *resultGpuTile2, *resultGpuTile3;
    cudaMalloc(&aGpu, rowsA * colsA * sizeof(double));
    cudaMalloc(&bGpu, rowsB * colsB * sizeof(double));
    cudaMalloc(&resultGpu, rowsA * colsB * sizeof(double));
    cudaMalloc(&resultGpuTile1, rowsA * colsB * sizeof(double));
    cudaMalloc(&resultGpuTile2, rowsA * colsB * sizeof(double));
    cudaMalloc(&resultGpuTile3, rowsA * colsB * sizeof(double));

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
    double kernelGemmStart = cpuSecond();
    gemm<<<grid,tpb>>>(aGpu, bGpu, resultGpu, rowsA, colsA, rowsB, colsB);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    double kernelGemmElapsed = cpuSecond() - kernelGemmStart;

    // TILE SIZE 1
    dim3 gridTile1((rowsA + TILE_SIZE_1 - 1) / TILE_SIZE_1, (colsB + TILE_SIZE_1 - 1) / TILE_SIZE_1, 1);
    dim3 tpbTile1(TILE_SIZE_1, TILE_SIZE_1, 1);
    double kernelGemmTile1Start = cpuSecond();
    int shmemTile1 = (2 * TILE_SIZE_1 * TILE_SIZE_1) * sizeof(double);
    tiled_gemm<<<gridTile1,tpbTile1, shmemTile1>>>(aGpu, bGpu, resultGpuTile1, rowsA, colsA, rowsB, colsB, TILE_SIZE_1, TILE_SIZE_1);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    double kernelGemmTile1Elapsed = cpuSecond() - kernelGemmTile1Start;

    // TILE SIZE 2
    dim3 gridTile2((rowsA + TILE_SIZE_2 - 1) / TILE_SIZE_2, (colsB + TILE_SIZE_2 - 1) / TILE_SIZE_2, 1);
    dim3 tpbTile2(TILE_SIZE_2, TILE_SIZE_2, 1);
    double kernelGemmTile2Start = cpuSecond();
    int shmemTile2 = (2 * TILE_SIZE_2 * TILE_SIZE_2) * sizeof(double);
    tiled_gemm<<<gridTile2, tpbTile2, shmemTile2>>>(aGpu, bGpu, resultGpuTile2, rowsA, colsA, rowsB, colsB, TILE_SIZE_2, TILE_SIZE_2);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    double kernelGemmTile2Elapsed = cpuSecond() - kernelGemmTile2Start;

    // TILE SIZE 3
    dim3 gridTile3((rowsA + TILE_SIZE_3 - 1) / TILE_SIZE_3, (colsB + TILE_SIZE_3 - 1) / TILE_SIZE_3, 1);
    dim3 tpbTile3(TILE_SIZE_3, TILE_SIZE_3, 1);
    double kernelGemmTile3Start = cpuSecond();
    int shmemTile3 = (2 * TILE_SIZE_3 * TILE_SIZE_3) * sizeof(double);
    tiled_gemm<<<gridTile3, tpbTile3, shmemTile3>>>(aGpu, bGpu, resultGpuTile3, rowsA, colsA, rowsB, colsB, TILE_SIZE_3, TILE_SIZE_3);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    double kernelGemmTile3Elapsed = cpuSecond() - kernelGemmTile3Start;

    // //@@ 7. Copy results from GPU to CPU
    cudaMemcpy(resultGpuCopy, resultGpu, rowsA * colsB * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(resultGpuCopyTile1, resultGpuTile1, rowsA * colsB * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(resultGpuCopyTile2, resultGpuTile2, rowsA * colsB * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(resultGpuCopyTile3, resultGpuTile3, rowsA * colsB * sizeof(double), cudaMemcpyDeviceToHost);

    // //@@ 8. Compare the results with the CPU reference result
    double cpuStart = cpuSecond();
    gemm_cpu(aCpu, bCpu, resultCpu, rowsA, colsA, rowsB, colsB);
    double cpuElapsed = cpuSecond() - cpuStart;

    if (printPretty == 1) {
        printf("Input matrix dim: (%d x %d) x (%d x %d)\n", rowsA, colsA, rowsB, colsB);

        printf("CPU Reference Result: \n");
        printMatrix(resultCpu, rowsA, colsB);
        printf("timing: %f ms\n", cpuElapsed * 1000);
        printf("-------------------------------------------------------------------------------------\n");
        
        printf("CUDA gemm result: \n");
        printMatrix(resultGpuCopy, rowsA, colsB);
        compareResult(resultGpuCopy, resultCpu, rowsA, colsB);
        printf("timing: %f ms\n", kernelGemmElapsed * 1000);
        printf("-------------------------------------------------------------------------------------\n");

        printf("CUDA tiled_gemm with tile (%d, %d) result: \n", TILE_SIZE_1, TILE_SIZE_1);
        printMatrix(resultGpuCopyTile1, rowsA, colsB);
        compareResult(resultGpuCopyTile1, resultCpu, rowsA, colsB);
        printf("timing: %f ms\n", kernelGemmTile1Elapsed * 1000);
        printf("-------------------------------------------------------------------------------------\n");

        printf("CUDA tiled_gemm with tile (%d, %d) result: \n", TILE_SIZE_2, TILE_SIZE_2);
        printMatrix(resultGpuCopyTile2, rowsA, colsB);
        compareResult(resultGpuCopyTile2, resultCpu, rowsA, colsB);
        printf("timing: %f ms\n", kernelGemmTile2Elapsed * 1000);
        printf("-------------------------------------------------------------------------------------\n");

        printf("CUDA tiled_gemm with tile (%d, %d) result: \n", TILE_SIZE_3, TILE_SIZE_3);
        printMatrix(resultGpuCopyTile3, rowsA, colsB);
        compareResult(resultGpuCopyTile3, resultCpu, rowsA, colsB);
        printf("timing: %f ms\n", kernelGemmTile3Elapsed * 1000);
        printf("-------------------------------------------------------------------------------------\n");
    } else {
        printf("cpu: %f ms\n", cpuElapsed * 1000);
        printf("gemm: %f ms\n", kernelGemmElapsed * 1000);
        printf("tiled_gemm(%dx%d): %f ms\n", TILE_SIZE_1, TILE_SIZE_1, kernelGemmTile1Elapsed * 1000);
        printf("tiled_gemm(%dx%d): %f ms\n", TILE_SIZE_2, TILE_SIZE_2, kernelGemmTile2Elapsed * 1000);
        printf("tiled_gemm(%dx%d): %f ms\n", TILE_SIZE_3, TILE_SIZE_3, kernelGemmTile3Elapsed * 1000);
    }



    // //@@ 9. Free host memory
    free(aCpu);
    free(bCpu);
    free(resultCpu);
    free(resultGpuCopy);
    free(resultGpuCopyTile1);
    free(resultGpuCopyTile2);
    free(resultGpuCopyTile3);

    // //@@ 10. Free device memory
    cudaFree(aGpu);
    cudaFree(bGpu);
    cudaFree(resultGpu);
    cudaFree(resultGpuTile1);
    cudaFree(resultGpuTile2);
    cudaFree(resultGpuTile3);

    return 0;
}
