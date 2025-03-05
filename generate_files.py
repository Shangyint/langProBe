import os

def create_benchmark_files(bench):
    # Create the benchmark directory
    benchmark_dir = f"langProBe/{bench}"
    os.makedirs(benchmark_dir, exist_ok=True)

    # Create the __init__.py file
    init_file = os.path.join(benchmark_dir, "__init__.py")
    with open(init_file, "w") as f:
        pass

    # Create the bench_data.py file
    data_file = os.path.join(benchmark_dir, f"{bench}_data.py")
    with open(data_file, "w") as f:
        pass

    # Create the bench_program.py file
    program_file = os.path.join(benchmark_dir, f"{bench}_program.py")
    with open(program_file, "w") as f:
        pass

    # Create the bench_utils.py file
    utils_file = os.path.join(benchmark_dir, f"{bench}_utils.py")
    with open(utils_file, "w") as f:
        pass


# Get benchmark name from first command line argument
import sys
benchmark_name = sys.argv[1]
create_benchmark_files(benchmark_name)