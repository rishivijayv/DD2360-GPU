all:
	nvcc -arch=sm_75 -o vecMatMultiply vecMatMultiply.cu
	nvcc -arch=sm_75 -o vecMatMultiplyFloat vecMatMultiplyFloat.cu

clean:
	rm -f vecMatMultiply vecMatMultiplyFloat
