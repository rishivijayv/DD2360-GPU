# Question 2
For all these instructions, we assume you are in the `q2/` directory. This directory has:
- `colab_notebook.ipynb`: Google Colab Notebook containig information for `ncu` profiling and stacked bar runtimes
- `Makefile`: Contains the compiling commands
- `plot.py`: Generates the plots for the stacked bar chart
- `README.md`: This file :) 
- `run_experiments.py`: A python file which runs the vector addition with different array sizes to get the runtimes to plot on the stacked bar chart
- `runtime_results.txt`: Output of `run_experiments.py` and contains the raw runtimes that are plotted in the stacked bar chart in the report. 
- `vecAddKernel.cu`: Contains the code for q2. 

## Compiling the Code
- Run the command `make`
- You will have a `vecAdd` binary

## Running the code
- Make sure the `vecAdd` binary exists
- Run `./vecAdd <N = vector length> <0/1 to compare results to CPU>`
    - For the second argument, passing `0` only runs the GPU execution and prints the runtimes HtoD transfer, DtoH transfer, and Kernel Execution
    - For the second argument, passing `1` runs the GPU execution and compares the results with the corresponding CPU result to ensure they are similar. 

## Runtimes for Stacked Bar Chart
- Make sure the `vecAdd` binary exists
- Delete `runtime_results.txt` (if it exists)
- Run `python3 run_experiments.py`
- The runtimes will be present in `runtime_results.txt` file. 
- To plot these runtimes, run `python3 plot.py`

We also have a Google Colab notebook `colab_notebook.ipynb` which includes our runs for the Nvidia Nsight profiling using `ncu` and the commands we used to get the runtimes for the stacked bar charts that are plotted in the report. 