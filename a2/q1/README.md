# Question 2
For all these instructions, we assume you are in the `q1/` directory. This directory has:
- `colab.ipynb`: Google Colab Notebook containig information for `ncu` profiling and other runs.
- `Makefile`: Contains the compiling commands
- `plot_histogram.py`: Generates the plots for the various histograms requested.
- `README.md`: This file :) 
- `generate_results.py`: Generates the results that can be plotted using `plot_histogram.py`.
- `histogram_<dist-type>_<input-length>.txt`: Contains the bins for `<input-length>` with the `<dist-type>` distribution. 
- `histogram.cu`: Contains the code for q1. 

## Compiling the Code
- Run the command `make`
- You will have a `histogram` binary

## Running the code
- Make sure the `histogram` binary exists
- Run `./histogram <inputLength> <0=uniform, 1=normal> <0=don't save histogram, 1=save histogram> <0=don't compare w/ cpu, 1=compare w/ cpu> <0..2=which kernel to use>`
    - For the 3rd argument, "saving" histogram means creating the a `histogram_<dist-type>_<input-length>.txt` file of the results
    - For the 5th argument, the different kernel types (0, 1, and 2) are as described in the report. 

## Runtimes for Histograms
- Make sure the `histogram` binary exists
- Delete all `histogram_<dist-type>_<input-length>.txt` files (if they exist)
- Run `python3 generate_results.py`
- The various `histogram_<dist-type>_<input-length>.txt` files will be generatd
- To plot these histograms, run `python3 plot_histogram.py`

We also have a Google Colab notebook `colab.ipynb` which includes our runs for the Nvidia Nsight profiling using `ncu` and the other runs that were performed and referred to in the report.

