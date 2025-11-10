import subprocess

# config of vector lengths to test with
# powers of 2 from 19 to 28
configs = [524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456]

output_file = f"runtime_results.txt"

with open(output_file, "w") as f_out:
    for config in configs:
        print(f"Running for size: {config}")
        f_out.write(str(config) + "\n")
        
        sum_host_to_device = 0.0
        sum_kernel_exec = 0.0
        sum_device_to_host = 0.0
        
        for run_num in range(3):
            print(f"Run {run_num}")
            result = subprocess.run([f"./vecAdd"] + [str(config)] + ["0"], capture_output=True, text=True)
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


        print(f"Averages: host_to_device={avg_host_to_device}s, kernel_exec={avg_kernel_exec}s, device_to_host={avg_device_to_host}s\n")
        print("---------------")
        
        f_out.write(f"host_to_device: {avg_host_to_device}\n")
        f_out.write(f"kernel_exec: {avg_kernel_exec}\n")
        f_out.write(f"device_to_host: {avg_device_to_host}\n")
        f_out.write("-----------------------\n")
