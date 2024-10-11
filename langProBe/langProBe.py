########################## Benchmarks ##########################
import importlib

benchmarks = [".gsm8k", ".humaneval", ".MATH"]

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


if __name__ == "__main__":
    for benchmark in benchmarks:
        register_benchmark(benchmark)

    for benchmark in registered_benchmarks:
        print(
            f"Registered benchmark: {benchmark.benchmark.__name__}, with programs: {[program.__name__ for program in benchmark.programs]}, with metric: {benchmark.metric}"
        )
