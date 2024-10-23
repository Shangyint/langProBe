########################## Benchmarks ##########################
import importlib


# To use registered benchmarks, do
# `benchmark.benchmark, benchmark.programs, benchmark.metric`
registered_benchmarks = []


def check_benchmark(benchmark):
    try:
        assert hasattr(benchmark, "benchmark")
    except AssertionError:
        return False
    return True


def register_benchmark(benchmark: str):
    # import the benchmark module
    benchmark_metas = importlib.import_module(benchmark, package="langProBe")
    if check_benchmark(benchmark_metas):
        registered_benchmarks.extend(benchmark_metas.benchmark)
    else:
        raise AssertionError(f"{benchmark} does not have the required attributes")
    return benchmark_metas.benchmark


def register_all_benchmarks(benchmarks):
    for benchmark in benchmarks:
        register_benchmark(benchmark)
    return registered_benchmarks
