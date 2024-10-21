import dspy.teleprompt
from langProBe.benchmark import BenchmarkMeta, EvaluateBench
from langProBe.optimizers import create_optimizer
from .register_benchmark import register_all_benchmarks, register_benchmark
import dspy


def evaluate(benchmark_meta: BenchmarkMeta, lm, rm, optimizer):
    benchmark = benchmark_meta.benchmark()
    print(f"Evaluating {benchmark}")
    for program in benchmark_meta.programs:
        print(f"Program: {program}")
        evaluate_bench = EvaluateBench(
            benchmark=benchmark,
            program=program(),
            metric=benchmark_meta.metric,
            optimizers=[create_optimizer(optimizer, benchmark_meta.metric)],
            num_threads=8,
        )
        evaluate_bench.evaluate(dspy_config={"lm": lm, "rm": rm})
        print(f"Results: {evaluate_bench.results}")


def evaluate_all(lm, rm, optimizer):
    benchmarks = register_all_benchmarks()
    for benchmark in benchmarks:
        evaluate(benchmark, lm, rm, optimizer)


if __name__ == "__main__":
    evaluate(
        benchmark_module=register_benchmark(".hotpotQA"),
        lm=dspy.OpenAI(model="gpt-4o-mini"),
        rm=dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts"),
        optimizer=dspy.teleprompt.bootstrap.BootstrapFewShot,
    )
