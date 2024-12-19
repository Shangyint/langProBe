from contextlib import contextmanager
from pathlib import Path
import os
import sys

from dotenv import load_dotenv
import dspy

from .benchmark import BenchmarkMeta, EvaluateBench
from .optimizers import create_optimizer
from .register_benchmark import register_all_benchmarks

load_dotenv()


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
    num_threads=8,
    suppress_dspy_output=True,
    file_path=None,
    dataset_mode=None,
    use_devset=False,
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
    optimizers = benchmark_meta.optimizers
    benchmark_name = benchmark.__class__.__name__

    num_threads = benchmark_meta.num_threads or num_threads
    print(f"Evaluating {benchmark_name}")
    print(f"Train set size: {len(benchmark.train_set)}")
    print(f"Validation set size: {len(benchmark.val_set)}")
    print(f"Test set size: {len(benchmark.test_set)}")

    optimizer_names = [optimizer.optimizer.__name__ for optimizer in optimizers]
    for i, optimizer in enumerate(optimizers):
        if "name" not in optimizer.langProBe_configs:
            optimizer.langProBe_configs["name"] = optimizer_names[i]

    if file_path:
        stats_file = os.path.join(file_path, f"{benchmark_name}.stat")
        Path(stats_file).parent.mkdir(parents=True, exist_ok=True)

        with open(stats_file, "w") as f:
            f.write(
                f"benchmark: {benchmark_name}\n"
                f"lm: {lm}\n"
                f"rm: {rm}\n"
                f"train_set_size: {len(benchmark.train_set)}\n"
                f"val_set_size: {len(benchmark.val_set)}\n"
                f"test_set_size: {len(benchmark.test_set)}\n"
                f"optimizers: {optimizer_names}\n"
                f"optimizer_configs: {optimizers}\n"
            )

    for program in benchmark_meta.program:
        print(f"Program: {program.__class__.__name__}")

        print(f"Optimizers: {'; '.join(optimizer_names)}")
        with suppress_output(suppress=suppress_dspy_output):
            evaluate_bench = EvaluateBench(
                benchmark=benchmark,
                program=program,
                metric=benchmark_meta.metric,
                optimizers=[
                    create_optimizer(
                        optimizer,
                        benchmark_meta.metric,
                        num_threads=num_threads,
                    )
                    for optimizer in optimizers
                ],
                num_threads=num_threads,
                use_devset=use_devset,
            )
            evaluate_bench.evaluate(dspy_config={"lm": lm, "rm": rm})
        print(f"Results: {evaluate_bench.results}")
        if file_path:
            Path(file_path).mkdir(parents=True, exist_ok=True)
            for evaluation_result in evaluate_bench.results:
                file_name = f"{evaluation_result.benchmark}_{evaluation_result.program}_{evaluation_result.optimizer}"
                if evaluation_result.optimizer:
                    optimizer_header = "optimizer,optimizer_cost,optimizer_input_tokens,optimizer_output_tokens"
                    optimizer_values = (
                        f"{evaluation_result.optimizer},{evaluation_result.optimizer_cost},"
                        f"{evaluation_result.optimizer_input_tokens},{evaluation_result.optimizer_output_tokens},"
                    )
                else:
                    optimizer_header = ""
                    optimizer_values = ""
                with open(os.path.join(file_path, f"{file_name}.txt"), "w") as f:
                    f.write(
                        f"score,cost,input_tokens,output_tokens,{optimizer_header}\n"
                    )
                    f.write(
                        f"{evaluation_result.score},{evaluation_result.cost},{evaluation_result.input_tokens},"
                        f"{evaluation_result.output_tokens},{optimizer_values}\n"
                    )
                if evaluation_result.optimizer:
                    evaluation_result.optimized_program.save(
                        os.path.join(file_path, f"{file_name}.model")
                    )


def evaluate_all(
    benchmarks,
    lm,
    rm,
    num_threads=8,
    suppress_dspy_output=True,
    file_path=None,
    dataset_mode=None,
    use_devset=False,
):
    benchmarks = register_all_benchmarks(benchmarks)
    for benchmark_meta in benchmarks:
        evaluate(
            benchmark_meta,
            lm,
            rm,
            num_threads,
            suppress_dspy_output,
            file_path,
            dataset_mode,
            use_devset,
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
        "--lm",
        help="The language model to use for evaluation",
        type=str,
        default="openai/gpt-4o-mini",
    )

    parser.add_argument(
        "--lm_api_base",
        help="The language model API base to use for evaluation",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--lm_api_key",
        help="The language model API key to use for evaluation",
        type=str,
        default="None",
    )

    parser.add_argument(
        "--benchmark_set",
        help="The benchmark set to evaluate. Options are full, nonagent, agent.",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--benchmark",
        help="The benchmark to evaluate. If not provided, all benchmarks will be evaluated. Providing this argument will override the benchmark_set argument.",
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

    parser.add_argument(
        "--file_path",
        help="The file path to save the evaluation results",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--num_threads",
        help="The number of threads to use for evaluation",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--dspy_cache_path",
        help="The cache path for dspy. THIS IS NOT IMPLEMENTED YET. NEED SUPPORT FROM DSPY.\
            Now please use DSPY_CACHEDIR=/path/to/cache python -m ...",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--use_devset",
        help="Whether to use the dev set for evaluation",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    suppress_dspy_output = args.suppress_dspy_output
    dataset_mode = args.dataset_mode

    lm = (
        dspy.LM(args.lm)
        if not args.lm_api_base
        else dspy.LM(args.lm, api_base=args.lm_api_base, api_key=args.lm_api_key)
    )
    rm = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")

    agent_benchmarks = [
        # ".AlfWorld",
        ".AppWorld",
    ]

    nonagent_benchmarks = [
        ".hover",
        ".Iris",
        ".IReRa",
        ".hotpotQA",
        ".MATH",
        ".gsm8k",
        ".RAGQAArenaTech",
        ".MMLU",
        ".swebenchAnnotation",
        ".scone",
        ".hotpotQA_conditional",
        ".HeartDisease",
        ".judgebench",
        ".humaneval",
    ]

    benchmarks = (
        agent_benchmarks + nonagent_benchmarks
        if args.benchmark_set == "full"
        else (
            agent_benchmarks
            if args.benchmark_set == "agent"
            else (
                nonagent_benchmarks
                if args.benchmark_set == "nonagent"
                else agent_benchmarks + nonagent_benchmarks
            )
        )
    )

    if args.benchmark:
        benchmarks = [f".{args.benchmark}"]

    # get current time to append to the file name
    import datetime

    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
    file_path = args.file_path or f"evaluation_{current_time}"
    evaluate_all(
        benchmarks,
        lm,
        rm,
        suppress_dspy_output=suppress_dspy_output,
        file_path=file_path,
        dataset_mode=dataset_mode,
        num_threads=args.num_threads,
        use_devset=args.use_devset,
    )
