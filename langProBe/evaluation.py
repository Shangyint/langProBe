from contextlib import contextmanager
import os
import sys
import dspy.teleprompt
from langProBe.benchmark import BenchmarkMeta, EvaluateBench
from langProBe.optimizers import create_optimizer, default_optimizers
from langProBe.visualization import plot_benchmark_results
from langProBe.register_benchmark import register_all_benchmarks
import dspy


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


@contextmanager
def suppress_output(suppress=True):
    if suppress:
        # Save the original streams
        original_stderr = sys.stderr
        original_stdout = sys.stdout

        # Redirect stderr and stdout to devnull
        sys.stderr = open(os.devnull, "w")
        sys.stdout = open(os.devnull, "w")

    try:
        yield
    finally:
        if suppress:
            # Restore the original streams
            sys.stderr.close()
            sys.stdout.close()
            sys.stderr = original_stderr
            sys.stdout = original_stdout


def evaluate(
    benchmark_meta: BenchmarkMeta,
    lm,
    rm,
    optimizers,
    num_threads=8,
    suppress_dspy_output=True,
    file_path=None,
    dataset_mode=None,
):
    """
    benchmark_meta: BenchmarkMeta object to evaluate
    lm: Language model to use, should be an instance of dspy.LM
    rm: Retrieval model to use
    optimizers: List[type(Teleprompter) | (type(Teleprompter), kwargs_for_compile)]
    """
    dataset_mode = dataset_mode or benchmark_meta.dataset_mode
    benchmark = benchmark_meta.benchmark(dataset_mode=dataset_mode)
    # Canonicalize optimizers to (optimizer, compile_kwargs) tuples
    optimizers = [
        optimizer
        if isinstance(optimizer, tuple)
        else (optimizer, {}, {}, dict(use_valset=False))
        for optimizer in optimizers
    ]
    print(f"Evaluating {benchmark.__class__.__name__}")
    for program in benchmark_meta.program:
        print(f"Program: {program.__class__.__name__}")
        optimizer_names = [optimizer[0].__name__ for optimizer in optimizers]
        print(f"Optimizers: {'; '.join(optimizer_names)}")
        with suppress_output(suppress=suppress_dspy_output):
            evaluate_bench = EvaluateBench(
                benchmark=benchmark,
                program=program,
                metric=benchmark_meta.metric,
                optimizers=[
                    (
                        create_optimizer(
                            optimizer[0],
                            benchmark_meta.metric,
                            optimizer[1],
                            optimizer[2],
                        ),
                        optimizer[3],
                    )
                    for optimizer in optimizers
                ],
                num_threads=num_threads,
            )
            evaluate_bench.evaluate(dspy_config={"lm": lm, "rm": rm})
        print(f"Results: {evaluate_bench.results}")
        if file_path:
            with open(file_path, "a") as f:
                result_list = []
                for scores in evaluate_bench.results.values():
                    if isinstance(scores, list):
                        result_list.extend(scores)
                    else:
                        result_list.append(scores)
                f.write(
                    f"{benchmark.__class__.__name__},{program.__name__},{','.join(optimizer_names)},{','.join(map(str, result_list))}\n"
                )


def evaluate_all(
    benchmarks,
    lm,
    rm,
    optimizers,
    num_threads=8,
    suppress_dspy_output=True,
    file_path=None,
    dataset_mode=None,
):
    benchmarks = register_all_benchmarks(benchmarks)
    for benchmark_meta in benchmarks:
        evaluate(
            benchmark_meta,
            lm,
            rm,
            optimizers,
            num_threads,
            suppress_dspy_output,
            file_path,
            dataset_mode,
        )


if __name__ == "__main__":
    # Allow to pass an arg suppress_dspy_output from the command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suppress_dspy_output",
        help="Whether to suppress dspy output",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--benchmark",
        help="The benchmark to evaluate. If not provided, all benchmarks will be evaluated.",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--dataset_mode",
        help="The dataset mode to use for evaluation. Options are: full, lite (500), tiny (200), test (20).\
        when not provided, the default dataset mode in BenchmarkMeta will be used.",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    suppress_dspy_output = args.suppress_dspy_output
    dataset_mode = args.dataset_mode

    optimizers = default_optimizers

    lm = dspy.LM("openai/gpt-4o-mini")
    rm = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")

    benchmarks = (
        [
            ".hover",
            ".AlfWorld",
            ".humaneval",
            ".Iris",
            ".IReRa",
            ".hotpotQA",
            ".MATH",
            ".gsm8k",
            ".AppWorld",
            ".RAGQAArenaTech",
            ".MMLU",
            ".swe_bench_verified_annotation_task",
            ".scone",
            ".hotpotQA_conditional",
        ]
        if not args.benchmark
        else [f".{args.benchmark}"]
    )
    # get current time to append to the file name
    import datetime

    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
    file_path = f"evaluation_{current_time}.csv"
    evaluate_all(
        benchmarks,
        lm,
        rm,
        optimizers,
        suppress_dspy_output=suppress_dspy_output,
        file_path=file_path,
        dataset_mode=dataset_mode,
    )

    plot_benchmark_results(file_path)
