# Question 3
For all these instructions, we assume you are in the `q3/` directory. This directory has:
- `colab_notebook.ipynb`: Google Colab Notebook containig information for `ncu` profiling and stacked bar runtimes
- `Makefile`: Contains the compiling commands
- `plot.py`: Generates the plots for the stacked bar charts
- `README.md`: This file :) 
- `run_experiments.py`: A python file which runs the matrix multiplication with different sizes to get the runtimes to plot on the stacked bar chart. Can be run with either the `double` or the `float` version of matrix mulitplication binaries.
- `runtime_results_vecMatMultiply.txt`: Output of `run_experiments.py` when running matrix multiplication using `double` as the operand and result data-type. 
- `runtime_results_vecMatMultiplyFloat.txt`: Output of `run_experiments.py` when running matrix multiplication using `float` as the operand and result data-type. 
- `vecMatMultiply.cu`: Contains the code for q3 where we use `double` as the operand and result data type.
- `vecMatMultiplyFloat.cu`: Contains the code for q3 where we use `float` as the operand and the result data type.


## Compiling the Code
- Run `make`
- You will have a `vecMatMultiply` and a `vecMatMultiplyFloat` binary. 

## Running the Code
The following instructions are the for `double` version of matrix-multiplication. The same steps can be replicated with the `float` version: just replace `vecMatMultiply` with `vecMatMultiplyFloat`. 
- Make sure `vecMatMultiply` exists
- Run `./vecMatMultiply <num_A_rows> <num_A_cols> <num_B_rows> <num_C_cols> <0/1 to compare results to CPU>`
    - Make sure that `<num_A_cols> = <num_B_rows>`
    - For the second argument, passing `0` only runs the GPU execution and prints the runtimes HtoD transfer, DtoH transfer, and Kernel Execution
    - For the second argument, passing `1` runs the GPU execution and compares the results with the corresponding CPU result to ensure they are similar (also prints the runtimes above).

## Runtimes for Stacked Bar Chart
The following instructions are the for `double` version of matrix-multiplication. The same steps can be replicated with the `float` version: just replace `vecMatMultiply` with `vecMatMultiplyFloat` and `runtime_results_vecMatMultiply.txt` with `runtime_results_vecMatMultiplyFloat.txt`. 
- Make sure `vecMatMultiply` exists
- Delete `runtime_results_vecMatMultiply.txt` (if it exists)
- Run `python3 run_experiments.py vecMatMultiply`
- The runtimes will be present in `runtime_results_vecMatMultiply.txt`
- To plot these runtimes, run `python3 plot.py runtime_results_vecMatMultiply.txt`.

The Google Colab notebook `colab_notebook.ipynb` includes our runs for the Nvidia Nsight profiling using `ncu` and the commands we used to get the runtimes for the stacked bar charts that are plotted in the report. 