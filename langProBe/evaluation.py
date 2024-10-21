import dspy.teleprompt
from langProBe.benchmark import BenchmarkMeta, EvaluateBench
from langProBe.optimizers import create_optimizer
from .register_benchmark import register_all_benchmarks, register_benchmark
import dspy

lm = dspy.LM("openai/gpt-4o-mini")


class CompareAnswerSignature(dspy.Signature):
    """
    Compare the answer to the ground truth answer.
    """

    answer = dspy.InputField(desc="The answer to a problem")
    ground_truth = dspy.InputField(desc="The ground truth answer to the same problem")
    is_correct = dspy.OutputField(
        desc="Whether the answer is correct, either True or False."
    )


class CompareAnswer(dspy.Module):
    def __init__(self):
        self.compare_answer = dspy.ChainOfThought(CompareAnswerSignature)

    def forward(self, ground_truth, answer):
        pred = self.compare_answer(answer=answer, ground_truth=ground_truth)
        return pred


def llm_as_judge_evaluate(gold, pred, extract_answer_fun=lambda x: x.answer):
    compare_answer = CompareAnswer()
    answer_raw = compare_answer(
        ground_truth=extract_answer_fun(gold), answer=extract_answer_fun(pred)
    ).is_correct
    if answer_raw.lower().startswith("true"):
        return True
    else:
        return False


def evaluate(benchmark_meta: BenchmarkMeta, lm, rm, optimizers, num_threads=8):
    """
    benchmark_meta: BenchmarkMeta object to evaluate
    lm: Language model to use, should be an instance of dspy.LM
    rm: Retrieval model to use
    optimizers: List[type(Teleprompter) | (type(Teleprompter), kwargs_for_compile)]
    """
    benchmark = benchmark_meta.benchmark(dataset_mode=benchmark_meta.dataset_mode)
    print(f"Evaluating {benchmark}")
    for program in benchmark_meta.program:
        print(f"Program: {program}")
        evaluate_bench = EvaluateBench(
            benchmark=benchmark,
            program=program(),
            metric=benchmark_meta.metric,
            optimizers=[
                create_optimizer(optimizer, benchmark_meta.metric)
                if isinstance(optimizer, type)
                else create_optimizer(
                    optimizer[0], benchmark_meta.metric, **optimizer[1]
                )
                for optimizer in optimizers
            ],
            num_threads=num_threads,
        )
        evaluate_bench.evaluate(dspy_config={"lm": lm, "rm": rm})
        print(f"Results: {evaluate_bench.results}")


def evaluate_all(lm, rm, optimizers):
    benchmarks = register_all_benchmarks()
    for benchmark in benchmarks:
        evaluate(benchmark, lm, rm, optimizers)


if __name__ == "__main__":
    evaluate(
        benchmark_meta=register_benchmark(".MATH")[0],
        lm=lm,
        rm=dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts"),
        optimizers=[
            dspy.teleprompt.BootstrapFewShot,
            dspy.teleprompt.BootstrapFewShotWithRandomSearch,
            (dspy.teleprompt.MIPROv2, {"requires_permission_to_run": False, "num_trials": 10}),
        ],
    )

    evaluate(
        benchmark_meta=register_benchmark(".hotpotQA")[0],
        lm=lm,
        rm=dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts"),
        optimizers=[
            dspy.teleprompt.BootstrapFewShot,
            dspy.teleprompt.BootstrapFewShotWithRandomSearch,
            (dspy.teleprompt.MIPROv2, {"requires_permission_to_run": False, "num_trials": 10}),
        ],
    )

    evaluate(
        benchmark_meta=register_benchmark(".gsm8k")[0],
        lm=lm,
        rm=dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts"),
        optimizers=[
            dspy.teleprompt.BootstrapFewShot,
            dspy.teleprompt.BootstrapFewShotWithRandomSearch,
            (dspy.teleprompt.MIPROv2, {"requires_permission_to_run": False, "num_trials": 10}),
        ],
    )

    evaluate(
        benchmark_meta=register_benchmark(".humaneval")[0],
        lm=lm,
        rm=dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts"),
        optimizers=[
            dspy.teleprompt.BootstrapFewShot,
            dspy.teleprompt.BootstrapFewShotWithRandomSearch,
            (dspy.teleprompt.MIPROv2, {"requires_permission_to_run": False, "num_trials": 10}),
        ],
    )
