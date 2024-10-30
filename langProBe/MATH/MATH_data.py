from ..benchmark import Benchmark
import dspy

from datasets import load_dataset


class MATHBench(Benchmark):
    def init_dataset(self):
        raw_datasets = load_dataset("lighteval/MATH", "all")
        self.dataset = [
            dspy.Example(**x).with_inputs("problem") for x in raw_datasets["train"]
        ]
        self.test_set = [
            dspy.Example(**x).with_inputs("problem") for x in raw_datasets["test"]
        ]
