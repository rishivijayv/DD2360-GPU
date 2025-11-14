import subprocess

# Sizes to use for histogram generation
element_counts = [1024, 10240, 102400, 1024000]
distributions = ["uniform", "normal"]

def get_distribution_number(dist):
    if dist == "uniform":
        return "0"
    else:
        return "1"


for distribution in distributions:
    print(f"Generating Graphs for {distribution.upper()} distribution")
    for element_count in element_counts:
        print(f"Current N={element_count}")
        output_file = f"histogram_{distribution}_{element_count}.txt"
        with open(output_file, "w") as f_out:
            command = ["./histogram", f"{element_count}", get_distribution_number(distribution), "1", "1", "2"]
            print(f"Running: {' '.join(command)}")
            result = subprocess.run(command, capture_output=True, text=True)
            print(result.stdout)
        print("-----------------------------------")
    print("------------------------------")
