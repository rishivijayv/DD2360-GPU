# Question 3
For all these instructions, we assume you are in the `q3/` directory. This directory has:
- `colab.ipynb`: Google Colab Notebook containig information for runs made on Google Colab
- `Makefile`: Contains the compiling commands
- `plot.py`: Generates the line-bar plots for the runtimes (on colab and on the school GPU cluster). Needs the name of the text file containing the runtimes to plot in a specific format.
- `README.md`: This file :) 
- `run_experiments.py`: A python file which runs the matrix multiplication with different sizes to get the runtimes to plot on the line-bar.
- `school_times.txt`: Output of `run_experiments.py` when running matrix multiplication on the school GPU cluster (H100 GPU). 
- `colab_times.txt`: Output of `run_experiments.py` when running matrix multiplication on Google Cola (Tesla T4).
- `vecMatMultiply.cu`: Contains the code for q3.


## Compiling the Code
- Run `make`
- You will have a `vecMatMultiply` binary. 

## Running the Code 
- Make sure `vecMatMultiply` exists
- Run `./vecMatMultiply <num_A_rows> <num_A_cols> <num_B_rows> <num_C_cols> <0=print_timing_for_graphing,1=print_user_friendly_output>`
    - Make sure that `<num_A_cols> = <num_B_rows>`
    - For the second argument, passing `0` returns a minimalist output that is used for graphing the runtimes
    - For the second argument, passing `1` prints the output in the format requested in the handout. Note that the entire result of the multiplication is not printed, but only the first 5 rows/cols are printed (if result has more than 5 rows/cols). The result of the comparison is printed however (as seen in the screenshots).

## Runtimes for Stacked Bar Chart 
- Make sure `vecMatMultiply` exists
- Run `python3 run_experiments.py`
- The runtimes will be present in `runtime_results.txt`
- To plot these runtimes, run `python3 plot.py runtime_results.txt`. You will first get a plot with CPU runtime, and then get a plot with only the GPU runtimes.

The Google Colab notebook `colab.ipynb` includes our runs for the Nvidia Nsight profiling using `ncu` and other runs requested on the handout that were made on the Google Colab.
