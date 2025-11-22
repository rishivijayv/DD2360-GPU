import matplotlib.pyplot as plt
import numpy as np
import sys

if len(sys.argv) != 2:
    print("Usage: python3 plot.py <name_of_results_file>.txt")
    exit(1)

RESULT_FILE = sys.argv[1]
configs = []
data_dict = {
    "cpu": [],
    "gemm": [],
    "tile_gemm(8x8)": [],
    "tile_gemm(16x16)": [],
    "tile_gemm(32x32)": [],
}

with open(RESULT_FILE) as f:
    lines = [line.strip() for line in f if line.strip() and not line.startswith("-")]
    
    for i in range(0, len(lines), 6):
        config = lines[i].split()[1]
        configs.append(config)
        data_dict["cpu"].append(float(lines[i+1].split()[1]))
        data_dict["gemm"].append(float(lines[i+2].split()[1]))
        data_dict["tile_gemm(8x8)"].append(float(lines[i+3].split()[1]))
        data_dict["tile_gemm(16x16)"].append(float(lines[i+4].split()[1]))
        data_dict["tile_gemm(32x32)"].append(float(lines[i+5].split()[1]))



plt.figure(figsize=(8,6))

# WITH CPU TIMES
for category, y_values in data_dict.items():
    plt.plot(configs, y_values, marker='o', label=category)

plt.xscale('log')
plt.yscale('log')

plt.xlabel("Matrix Sizes")
plt.ylabel("Runtime (ms)")
plt.title("Log-Log Chart of Matrix-Sizes and Runtimes(ms)")
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)

plt.tight_layout()
plt.show()

# WITHOUT CPU TIMES
without_cpu = {k: v for k, v in data_dict.items() if k != "cpu"}
for category, y_values in without_cpu.items():
    plt.plot(configs, y_values, marker='o', label=category)

plt.xscale('log')
plt.yscale('log')

plt.xlabel("Matrix Sizes")
plt.ylabel("Runtime (ms)")
plt.title("Log-Log Chart of Matrix-Sizes and Runtimes(ms) Without CPU")
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)

plt.tight_layout()
plt.show()
