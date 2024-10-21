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
