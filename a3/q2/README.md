# Question 2
For all these instructions, we assume you are in the `q2/` directory. This directory has:
- `Makefile`: Contains the compiling commands
- `normal_v_streams.txt`: Contains the raw data plotted for comparing the runtime of streamed version and the normal version of addition on the GPU (keeping segment size constant)
- `normal_vs_streams.py`: The python file which runs the experiment to generate the data in the `normal_v_streams.txt` file (ie, part1)
- `plot.py`: Generates the plots for part1 and part3 of Q2
- `README.md`: This file :) 
- `segments.py`: The python file which runs the experiments to generate the data for the impact of segment size on the streamed version of vector addition (keeping the vector length constant) -- ie, part3
- `segments.txt`: Contains the raw data plotted for comparing the impact of segment size on the performance of the streamed vector addition version
- `streamed_vec_add.nsys-rep`: Contains a trace obtained using `nsys` on the H100 GPU on the School's Cluster. Can be visualized using the NVIDIA Nsight Systems program.
- `vecAddKernel.cu`: Contains the code for q2.

## Compiling the Code
- Run the command `make`
- You will have a `vecAdd` binary

## Running the code
- Make sure the `vecAdd` binary exists
- Run `./vecAdd <N = vector length> <segment length> <0=don't check cpu with gpu, 1=check cpu with gpu> <0=don't do normal vector addition, 1=also do normal vector addition>`
    - For the third argument, passing `0` only runs the GPU execution (streamed version is always run, whether or not normal GPU vector addition run depends on fourth argument)
    - For the fourth argument, passing `1` runs the normal vector addition on GPU as well (ie, the non-streamed version from A1)
    - In all cases, the run times (in ms) of the GPU vector additions that are run are printed. If it is compared to the CPU version, then the result of the comparison is printed too (a message saying all is good if everything is within range, or the indices where a difference was found) 

## Runtimes for Normal vs Streamed GPU Vector Addition (part1)
- Make sure the `vecAdd` binary exists
- Delete `normal_v_streams.txt` (if it exists)
- Run `python3 normal_vs_streams.py`
- The runtimes will be present in `normal_v_streams.txt` file. 
- To plot these runtimes, run `python3 plot.py q1`

## Runtimes for Varying Segment Length in Streamed Vector Addition (part3)
- Make sure the `vecAdd` binary exists
- Delete `segments.txt` (if it exists)
- Run `python3 segments.py`
- The runtimes will be present in `segments.txt` file. 
- To plot these runtimes, run `python3 plot.py q3`