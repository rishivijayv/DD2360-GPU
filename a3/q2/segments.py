import subprocess

# Matrix sizes to use for the testing
configs = [
    512 * 10**i for i in range(0, 5)
]

output_file = f"segments.txt"
vec_len = 10240000

with open(output_file, "w") as f_out:
    for config in configs:
        print(f"Running for vector length: ", vec_len, "and segment size: ", config)
        f_out.write(f"{config}" + "\n")
        
        sum_streams = 0.0

        for run_num in range(3):
            print(f"Run num: {run_num}")
            result = subprocess.run([f"./vecAdd"] + [str(vec_len)] + [str(config)] + ["0"] + ["0"], capture_output=True, text=True)
            output = result.stdout
            print(output)

            for line in output.splitlines():
                if line.startswith("streams"):
                    sum_streams += float(line.split()[1])
        
        avg_streams = sum_streams / 3

        print(f"Averages: streams={avg_streams:.3f}ms\n")
        print("-----------------------------------------------------------------------------------------------------------")
        f_out.write(f"streams: {avg_streams}\n")
        f_out.write("-----------------------\n")
