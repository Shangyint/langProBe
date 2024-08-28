from .humaneval_program import NaiveCodeGenerator
from .humaneval_utils import human_eval_evaluate
from ..benchmark import Benchmark, EvaluateBench
from datasets import load_dataset
import dspy


class HumanEvalBench(Benchmark):
    def init_dataset(self):
        raw_datasets = load_dataset("openai_humaneval")["test"]
        self.dataset = [
            dspy.Example(**x).with_inputs(
                "prompt", "test", "entry_point", "canonical_solution", "task_id"
            )
            for x in raw_datasets
        ]

    def create_splits(self):
        self.train_set, self.dev_set, self.test_set = (
            self.dataset[:20],
            self.dataset,
            self.dataset,
        )


human_eval_bench = HumanEvalBench()
evaluate_naive_program = EvaluateBench(
    human_eval_bench, NaiveCodeGenerator(), human_eval_evaluate
)

with dspy.context(lm=dspy.OpenAI(model="gpt-3.5-turbo", max_tokens=1000)):
    evaluate_naive_program.evaluate()
