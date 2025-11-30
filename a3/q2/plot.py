import matplotlib.pyplot as plt
import numpy as np
import sys

if len(sys.argv) != 2:
    print("Usage: python3 plot.py <q1 or q3>")
    exit(1)

QUESTION = 1 if sys.argv[1] == "q1" else 3
RESULT_FILE = "normal_v_streams.txt" if QUESTION == 1 else "segments.txt"
X_LABEL = "Vector Length (n)" if QUESTION == 1 else "Segment Size"
TITLE = "Log-Log Plot for Normal vs Streamed GPU Vector Addition (S_seg = 102400)" if QUESTION == 1 else "Log-Log Plot for Impact of Segment Length on Runtime (n = 10240000)"
configs = []
data_dict = {
    "streams": [],
    "normal": []
} if QUESTION == 1 else {
    "streams": []
}

with open(RESULT_FILE) as f:
    lines = [line.strip() for line in f if line.strip() and not line.startswith("-")]
    increment = 3 if QUESTION == 1 else 2

    for i in range(0, len(lines), increment):
        config = lines[i]
        configs.append(config)
        if QUESTION == 1:
            data_dict["normal"].append(float(lines[i+1].split()[1]))
            data_dict["streams"].append(float(lines[i+2].split()[1]))            
        else:
            data_dict["streams"].append(float(lines[i+1].split()[1]))


configs = [int(value) for value in configs]

plt.figure(figsize=(8,6))

for category, y_values in data_dict.items():
    plt.plot(configs, y_values, marker='o', label=category)

plt.xscale('log')
plt.yscale('log')

plt.xlabel(X_LABEL)
plt.ylabel("Runtime (ms)")
plt.title(TITLE)
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)

plt.tight_layout()
plt.show()