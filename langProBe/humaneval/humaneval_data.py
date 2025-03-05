from ..benchmark import Benchmark
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

        self.test_set = self.dataset[len(self.dataset) // 2 :]
        self.dataset = self.dataset[: len(self.dataset) // 2]

        self.train_set = self.dataset[:15]
        self.val_set = self.dataset[15:]
        self.dev_set = self.dataset
