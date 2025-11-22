import subprocess

# Matrix sizes to use for the testing
configs = [
    ["16", "16", "16", "16"],
    ["32", "32", "32", "32"],
    ["64", "64", "64", "64"],
    ["128", "128", "128", "128"],
    ["256", "256", "256", "256"],
    ["512", "512", "512", "512"],
    ["1024", "1024", "1024", "1024"],
    ["2048", "2048", "2048", "2048"]
]

output_file = f"runtime_results.txt"

with open(output_file, "w") as f_out:
    for config in configs:
        print(f"Running for config: ", config)
        f_out.write(" ".join(config) + "\n")
        
        sum_cpu = sum_gemm = sum_tiled_gemm_8 = sum_tiled_gemm_16 = sum_tiled_gemm_32 = 0.0

        for run_num in range(3):
            print(f"Run num: {run_num}")
            result = subprocess.run([f"./vecMatMultiply"] + config + ["0"], capture_output=True, text=True)
            output = result.stdout
            print(output)

            for line in output.splitlines():
                if line.startswith("cpu"):
                    sum_cpu += float(line.split()[1])
                elif line.startswith("gemm"):
                    sum_gemm += float(line.split()[1])
                elif line.startswith("tiled_gemm(8x8)"):
                    sum_tiled_gemm_8 += float(line.split()[1])
                elif line.startswith("tiled_gemm(16x16)"):
                    sum_tiled_gemm_16 += float(line.split()[1])
                elif line.startswith("tiled_gemm(32x32)"):
                    sum_tiled_gemm_32 += float(line.split()[1])
        
        avg_cpu = sum_cpu / 3
        avg_gemm = sum_gemm / 3
        avg_tiled_gemm_8 = sum_tiled_gemm_8 / 3
        avg_tiled_gemm_16 = sum_tiled_gemm_16 / 3
        avg_tiled_gemm_32 = sum_tiled_gemm_32 / 3

        print(f"Averages: cpu={avg_cpu:.3f}ms, gemm={avg_gemm:.3f}ms, tiled(8x8)={avg_tiled_gemm_8:.3f}ms, tiled(16x16)={avg_tiled_gemm_16:.3f}ms, tiled(32x32)={avg_tiled_gemm_32:.3f}ms\n")
        print("-----------------------------------------------------------------------------------------------------------")
        
        f_out.write(f"cpu: {avg_cpu}\n")
        f_out.write(f"gemm: {avg_gemm}\n")
        f_out.write(f"tiled_gemm(8x8): {avg_tiled_gemm_8}\n")
        f_out.write(f"tiled_gemm(16x16): {avg_tiled_gemm_16}\n")
        f_out.write(f"tiled_gemm(32x32): {avg_tiled_gemm_32}\n")
        f_out.write("-----------------------\n")
