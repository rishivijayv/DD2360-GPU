import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

if len(sys.argv) != 2:
    print("Usage: python3 plot.py <name_of_results_file>.txt")
    exit(1)

RESULT_FILE = sys.argv[1]
data_dict = {
    "init": [],
    "cpu": [],
    "copy_data": [],
    "gpu_naive": [],
    "gpu_tiling_block": [],
    "gpu_tiling_thread": []
}

df = pd.read_csv(RESULT_FILE, delimiter=';')
print(df)

data_dict["init"] = list(df['init'])
data_dict["cpu"] = list(df['cpu'])
data_dict["copy_data"] = list(df['copy_data'])
data_dict["gpu_naive"] = list(df['gpu_naive'])
data_dict["gpu_tiling_block"] = list(df['gpu_tiling_block'])
data_dict["gpu_tiling_thread"] = list(df['gpu_tiling_thread'])
configs = list(df['size'])
configs = [2 ** it for it in configs]


plt.figure(figsize=(8,6))

# WITH CPU TIMES
for category, y_values in data_dict.items():
    plt.plot(configs, y_values, marker='o', label=category)

plt.xscale('log')
plt.yscale('log')

plt.xlabel("Input size")
plt.ylabel("Runtime (ms)")
plt.title("Log-Log Chart of Array-Sizes and Runtimes(ms)")
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)

plt.tight_layout()
plt.show()

