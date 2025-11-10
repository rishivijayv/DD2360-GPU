import matplotlib.pyplot as plt
import numpy as np
import sys

if len(sys.argv) != 2:
    print("Usage: python3 plot.py <name_of_results_file>.txt")
    exit(1)

RESULT_FILE = sys.argv[1]
configs = []
host_to_device_times = []
kernel_exec_times = []
device_to_host_times = []

with open(RESULT_FILE) as f:
    lines = [line.strip() for line in f if line.strip() and not line.startswith("-")]
    
    for i in range(0, len(lines), 4):
        config = lines[i]
        configs.append(config)
        host_to_device_times.append(float(lines[i+1].split()[1]))
        kernel_exec_times.append(float(lines[i+2].split()[1]))
        device_to_host_times.append(float(lines[i+3].split()[1]))


x = np.arange(len(configs))
width = 0.6  # width of bars

# Create stacked bars
plt.bar(x, host_to_device_times, width, label="host_to_device")
plt.bar(x, kernel_exec_times, width, bottom=host_to_device_times, label="kernel_exec")
bottoms = np.array(host_to_device_times) + np.array(kernel_exec_times)
plt.bar(x, device_to_host_times, width, bottom=bottoms, label="device_to_host")

# Labels and title
plt.xticks(x, configs, rotation=45, ha="right")
plt.ylabel("Time (s)")
plt.xlabel("Matrix Configurations")
plt.title("Stacked Bar Chart of Runtimes")
plt.legend()

plt.tight_layout()
plt.show()
