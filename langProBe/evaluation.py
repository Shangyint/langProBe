import dspy.teleprompt
from langProBe.benchmark import EvaluateBench
from langProBe.optimizers import create_optimizer
from .register_benchmark import register_all_benchmarks
import dspy


def evaluate(lm, rm, optimizer):
    benchmarks = register_all_benchmarks()
    for benchmark in benchmarks:
        initialized_benchmark = benchmark.benchmark()
        print(f"Evaluating {initialized_benchmark}")
        for program in benchmark.programs:
            print(f"Program: {program}")
            evaluate_bench = EvaluateBench(
                benchmark=initialized_benchmark,
                program=program(),
                metric=benchmark.metric,
                optimizers=[create_optimizer(optimizer, benchmark.metric)],
                num_threads=8,
            )
            evaluate_bench.evaluate(dspy_config={"lm": lm, "rm": rm})
            print(f"Results: {evaluate_bench.results}")


if __name__ == "__main__":
    evaluate(
        lm=dspy.OpenAI(model="gpt-4o-mini"),
        rm=dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts"),
        optimizer=dspy.teleprompt.bootstrap.BootstrapFewShot,
    )
