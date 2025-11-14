
#include <stdio.h>
#include <sys/time.h>
#include <random>
#include <algorithm>
#include <cmath>
#include <string>

#define BIN_LIMIT 127
#define NUM_BINS 4096
#define THREADS_PER_BLOCK 1024

// Constants for distribution type
#define UNIFORM 0
#define NORMAL 1

// The CPU Timer
double cpuSecond() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

__global__ void histogram_kernel_v0(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {
  //@@ Insert code below to compute histogram of input using shared memory and atomics
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_elements) {
    return;
  }

  atomicAdd(&(bins[input[idx]]), 1);
}

__global__ void histogram_kernel_v1(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {
  //@@ Insert code below to compute histogram of input using shared memory and atomics
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_elements) {
    return;
  }


  // Threads will compute a "local" histogram on their part of the shared memory. 
  // Then this will be added to the "global" histogram.
  __shared__ unsigned int localBins[NUM_BINS];

  // Initialize shared memory to 0
  if (threadIdx.x == 0) {
    for(int i = 0; i < NUM_BINS; i++) {
      localBins[i] = 0;
    }
  }
  // wait for initialization
  __syncthreads();

  // Each thread responsible for one element
  atomicAdd(&(localBins[input[idx]]), 1);
  // Wait for all threads to finish with their computation
  __syncthreads();

  // Get local bin back to global bin
  if (threadIdx.x == 0) {
    for (int i = 0; i < NUM_BINS; i++) {
      atomicAdd(&(bins[i]), localBins[i]);
    }
  }
}

__global__ void histogram_kernel_v2(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {
  //@@ Insert code below to compute histogram of input using shared memory and atomics
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Threads will compute a "local" histogram on their part of the shared memory. 
  // Then this will be added to the "global" histogram.
  __shared__ unsigned int localBins[NUM_BINS];

  // Initialize shared memory to 0
  // Each thread initializes 4 elements to 0. tidx 0 -> 0, 1024, 2048, 3072
  for (int i = 0; i < 4; i++) {
    int currIdx = threadIdx.x + i * blockDim.x;
    localBins[currIdx] = 0;
  }
  // wait for initialization
  __syncthreads();

  // Each thread responsible for one element
  // Only do this if the thread is within global array range
  if (idx < num_elements) {
    atomicAdd(&(localBins[input[idx]]), 1);
  }
  // Wait for all threads to finish with their computation
  __syncthreads();

  // Get local bin back to global bin
  for (int i = 0; i < 4; i++) {
      int currIdx = threadIdx.x + i * blockDim.x;
      atomicAdd(&(bins[currIdx]), localBins[currIdx]);
  }
}

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {
  //@@ Insert code below to clean up bins that saturate at 127
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_bins) {
    return;
  }

  if (bins[idx] > BIN_LIMIT) {
    bins[idx] = BIN_LIMIT;
  }
}

void histogram_cpu(unsigned int *input, unsigned int *bins, unsigned int len) {
  // We assume that the input is always between 0 and NUM_BINS - 1, so that the array
  // indexing will always work. Additionally, we assume that resultRef has already been
  // initialized to 0
  for (int i = 0; i < len; i++) {
    if (bins[input[i]] < BIN_LIMIT) {
      bins[input[i]] += 1;
    }
  }
}

// Initialize bins to 0 count
void initialize_bins(unsigned int *bins) {
  for (int i = 0; i < NUM_BINS; i++) {
    bins[i] = 0;
  }
}

void initialize_uniform_distribution(unsigned int *input, unsigned int len) {
  std::mt19937 gen(42);
  std::uniform_int_distribution<> dist{0, NUM_BINS - 1};

  for (int i = 0; i < len; i++) {
    input[i] = dist(gen);
  }
}

int compare_results(unsigned int *resultsGpu, unsigned int *resultsCpu) {
  int allOk = 1;

  for (int i = 0; i < NUM_BINS; i++) {
    if(resultsCpu[i] != resultsGpu[i]) {
      allOk = 0;
      printf("Difference at index %d: GPU[%d] = %d, CPU[%d] = %d\n", i, i, resultsGpu[i], i, resultsCpu[i]);
    }
  }

  if (allOk == 1) {
    printf("All results are same between GPU and CPU! :) \n");
  }

  return allOk;
}

void save_histogram(unsigned int *bins, int distributionType, int inputLength) {
    std::string distString;
    if (distributionType == UNIFORM) {
      distString = "uniform";
    } else {
      distString = "normal";
    }

    std::string filename = "histogram_" + distString + "_" + std::to_string(inputLength) + ".txt";
    FILE *fp = fopen(filename.c_str(), "w");
    if (!fp) {
        perror("Failed to open file to write");
        return;
    }

    // Number of bins in first line
    fprintf(fp, "%d\n", NUM_BINS);

    // bin[i] in each line
    for (int i = 0; i < NUM_BINS; i++) {
        fprintf(fp, "%d\n", bins[i]);
    }

    fclose(fp);
}

void initialize_normal_distribution(unsigned int *input, unsigned int len) {
  // Mean set to 4095 / 2 = 2047.5.
  // We want most of our datat to be between [0, 4095]. 
  // According to 68-95-99.7 rule (https://www.freecodecamp.org/news/normal-distribution-explained/), 
  // This means that we want 0 and 4095 to be 3s.d. away from mean of 2048. 
  // Thus, s.d. = 2047.5 / 3 ~ 682.5. We will clamp values that are less than 0 or greater than 4095 (ie, attribute
  // them to 0 or 4095 if they are smaller or greater, respectively).   
  std::mt19937 gen{42};
  std::normal_distribution dist{2047.5, 682.5};
  
  for (int i = 0; i < len; i++) {
    int generated = std::lround(dist(gen));
    if (0 <= generated && generated <= NUM_BINS - 1) {
      input[i] = generated;
    } else if (generated < 0) {
      input[i] = 0;
    } else {
      input[i] = NUM_BINS - 1;
    }
  }
}


int main(int argc, char **argv) {
  if (argc != 6) {
    fprintf(stderr, "Usage: %s <inputLength> <0=uniform, 1=normal> <0=don't save histogram, 1=save histogram> <0=don't compare w/ cpu, 1=compare w/ cpu> <0..2=which kernel to use>\n", argv[0]);
    exit(1);
  }

  //@@ Insert code below to read in inputLength from args

  int inputLength = atoi(argv[1]);
  int distributionType = atoi(argv[2]);
  int saveHistogram = atoi(argv[3]);
  int compareCpu = atoi(argv[4]);
  int kernelVersion = atoi(argv[5]);
  printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output
  unsigned int *hostInput = (unsigned int *)malloc(inputLength * sizeof(unsigned int));
  unsigned int *hostBins = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
  unsigned int *deviceResult = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
  
  //@@ Insert code below to initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)
  if (distributionType == UNIFORM) {
    initialize_uniform_distribution(hostInput, inputLength);
  } else {
    // assume distributionType is 1 and use Normal
    initialize_normal_distribution(hostInput, inputLength);
  }
  initialize_bins(hostBins);

  double cpuTimeStart, cpuTimeElapsed;

  if (compareCpu) {
    cpuTimeStart = cpuSecond();
    //@@ Insert code below to create reference result in CPU
    histogram_cpu(hostInput, hostBins, inputLength);
    cpuTimeElapsed = cpuSecond() - cpuTimeStart;
  }


  //@@ Insert code below to allocate GPU memory here
  unsigned int *deviceInput, *deviceBins;
  cudaMalloc(&deviceInput, inputLength * sizeof(unsigned int));
  cudaMalloc(&deviceBins, NUM_BINS * sizeof(unsigned int));

  double hostToDeviceTimeStart = cpuSecond();
  //@@ Insert code to Copy memory to the GPU here
  cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(unsigned int), cudaMemcpyHostToDevice);

  //@@ Insert code to initialize GPU results
  cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int));
  double hostToDeviceTimeElapsed = cpuSecond() - hostToDeviceTimeStart;

  //@@ Initialize the grid and block dimensions here
  dim3 gridFirst((inputLength + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1, 1);
  dim3 tpbFirst(THREADS_PER_BLOCK, 1, 1);

  double kernelTimeStart = cpuSecond();
  //@@ Launch the GPU Kernel here
  if (kernelVersion == 0) {
    histogram_kernel_v0<<<gridFirst, tpbFirst>>>(deviceInput, deviceBins, inputLength, NUM_BINS);
  } else if (kernelVersion == 1) {
      histogram_kernel_v1<<<gridFirst, tpbFirst>>>(deviceInput, deviceBins, inputLength, NUM_BINS);
  } else {
    // Assume kernelVersion is 2
    histogram_kernel_v2<<<gridFirst, tpbFirst>>>(deviceInput, deviceBins, inputLength, NUM_BINS);
  }
  cudaDeviceSynchronize();

  //@@ Initialize the second grid and block dimensions here
  dim3 gridCleaning((NUM_BINS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1, 1);
  dim3 tpbCleaning(THREADS_PER_BLOCK, 1, 1);

  //@@ Launch the second GPU Kernel here
  convert_kernel<<<gridCleaning, tpbCleaning>>>(deviceBins, NUM_BINS);
  cudaDeviceSynchronize();
  double kernelTimeElapsed = cpuSecond() - kernelTimeStart;

  double deviceToHostTimeStart = cpuSecond();
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(deviceResult, deviceBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
  double deviceToHostTimeElapsed = cpuSecond() - deviceToHostTimeStart;

  //@@ Insert code below to compare the output with the reference
  if (compareCpu) {
    int resultsEqual = compare_results(deviceResult, hostBins);

    if (resultsEqual && saveHistogram) {
      // Since the results are the same, can save any of them in the file
      save_histogram(hostBins, distributionType, inputLength);
      printf("Saved Histogram!\n");
    }
  }

  //@@ Free the GPU memory here
  cudaFree(deviceInput);
  cudaFree(deviceBins);

  //@@ Free the CPU memory here
  free(hostInput);
  free(hostBins);
  free(deviceResult);


  if (compareCpu) {
    printf("CPU Time: %f ms\n", cpuTimeElapsed * 1000); 
  }
  printf("GPU h-to-d time: %f ms\n", hostToDeviceTimeElapsed * 1000);
  printf("GPU kernel time: %f ms\n", kernelTimeElapsed * 1000);
  printf("GPU t-to-h time: %f ms\n", deviceToHostTimeElapsed * 1000);
  printf("GPU transfer and kernel time: %f ms\n", (hostToDeviceTimeElapsed + kernelTimeElapsed + deviceToHostTimeElapsed) * 1000);

  return 0;
}

