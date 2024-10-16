########################## Benchmarks ##########################
import importlib

benchmarks = [".gsm8k", ".MATH", ".hotpotQA", ".humaneval"]

# To use registered benchmarks, do
# `benchmark.benchmark, benchmark.programs, benchmark.metric`
registered_benchmarks = []


def check_benchmark(benchmark):
    try:
        assert hasattr(benchmark, "benchmark")
        assert hasattr(benchmark, "programs")
        assert hasattr(benchmark, "metric")
    except AssertionError:
        return False
    return True


def register_benchmark(benchmark: str):
    # import the benchmark module
    benchmark_module = importlib.import_module(benchmark, package="langProBe")
    if check_benchmark(benchmark_module):
        registered_benchmarks.append(benchmark_module)
    else:
        raise AssertionError(f"{benchmark} does not have the required attributes")


def register_all_benchmarks(benchmarks=benchmarks):
    for benchmark in benchmarks:
        register_benchmark(benchmark)
    return registered_benchmarks
