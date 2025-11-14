import numpy as np
import matplotlib.pyplot as plt
import glob


histogram_files = glob.glob("histogram_*.txt")


for file in sorted(histogram_files):
    with open(file) as f:
        length = int(f.readline())
        hist = np.array([int(line.strip()) for line in f])


    plt.bar(np.arange(length), hist, width=1.0)
    plt.xlabel("Bin")
    plt.ylabel("Count")
    plt.title(file.split(".")[0])
    plt.show()