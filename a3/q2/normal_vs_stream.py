import subprocess

# Matrix sizes to use for the testing
configs = [
    1024000 * 2**i for i in range(0, 6)
]

output_file = f"normal_v_streams.txt"
vec_len_start = 1024000
segment_size = 102400

with open(output_file, "w") as f_out:
    for config in configs:
        print(f"Running for vector length: ", config, "and segment size: ", segment_size)
        f_out.write(f"{config}" + "\n")
        
        sum_normal = sum_streams = 0.0

        for run_num in range(3):
            print(f"Run num: {run_num}")
            result = subprocess.run([f"./vecAdd"] + [str(config)] + [str(segment_size)] + ["0"] + ["1"], capture_output=True, text=True)
            output = result.stdout
            print(output)

            for line in output.splitlines():
                if line.startswith("normal"):
                    sum_normal += float(line.split()[1])
                elif line.startswith("streams"):
                    sum_streams += float(line.split()[1])
        
        avg_normal = sum_normal / 3
        avg_streams = sum_streams / 3

        print(f"Averages: normal={avg_normal:.3f}ms, streams={avg_streams:.3f}ms\n")
        print("-----------------------------------------------------------------------------------------------------------")
        
        f_out.write(f"normal: {avg_normal}\n")
        f_out.write(f"streams: {avg_streams}\n")
        f_out.write("-----------------------\n")
