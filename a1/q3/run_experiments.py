import subprocess
import sys

# Matrix sizes to use for the testing
configs = [
    ["1200", "8900", "8900", "9000"],
    ["1200", "17800", "17800", "9000"],
    ["1200", "35600", "35600", "9000"],
    ["1200", "71200", "71200", "9000"]
]

if len(sys.argv) != 2:
    print("Usage: python3 run_experiments.py <name_of_binary>")
    exit(1)

output_file = f"runtime_results_{sys.argv[1]}.txt"

with open(output_file, "w") as f_out:
    for config in configs:
        print(f"Running for config: ", config)
        f_out.write(" ".join(config) + "\n")
        
        sum_host_to_device = 0.0
        sum_kernel_exec = 0.0
        sum_device_to_host = 0.0
        
        for run_num in range(3):
            print(f"Run num: {run_num}")
            result = subprocess.run([f"./{sys.argv[1]}"] + config + ["0"], capture_output=True, text=True)
            output = result.stdout
            print(output)

            for line in output.splitlines():
                if line.startswith("host_to_device"):
                    sum_host_to_device += float(line.split()[1])
                elif line.startswith("kernel_exec"):
                    sum_kernel_exec += float(line.split()[1])
                elif line.startswith("device_to_host"):
                    sum_device_to_host += float(line.split()[1])
        
        avg_host_to_device = sum_host_to_device / 3
        avg_kernel_exec = sum_kernel_exec / 3
        avg_device_to_host = sum_device_to_host / 3


        print(f"Averages: host_to_device={avg_host_to_device:.3f}s, kernel_exec={avg_kernel_exec:.3f}s, device_to_host={avg_device_to_host:.3f}s\n")
        print("---------------")
        
        f_out.write(f"host_to_device: {avg_host_to_device:.3f}\n")
        f_out.write(f"kernel_exec: {avg_kernel_exec:.3f}\n")
        f_out.write(f"device_to_host: {avg_device_to_host:.3f}\n")
        f_out.write("-----------------------\n")
