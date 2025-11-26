# Question 2
For all these instructions, we assume you are in the `q2/` directory. This directory has:
- `colab.ipynb`: Google Colab Notebook containig information for runs made on Google Colab
- `Makefile`: Contains the compiling commands
- `plot.py`: Generates the line-bar plots for the runtimes. Needs the name of the text file containing the runtimes to plot in a specific format (in this case colab_runtimes.csv)
- `README.md`: This file :)
- `colab_runtimes.csv`: Runtimes when benchmarking for different grid sizes on goole colab
- `reduction.cu`: Contains the code for q3.
- `reduction_ncu.cu`: Reduced version of the code we used for ncu profiling

## Compiling the Code
- Run `make`
- You will have a `reduction` binary. 

## Running the Code 
- Make sure `reduction ` exists
- Run `./reduction <numElements>` or `./reduction` to run with set of preconfigured grid sizes
    - input needs to be big enough, e.g. >= for code working correctly

