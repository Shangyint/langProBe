from ..benchmark import Benchmark
import dspy

from datasets import load_dataset


class GSM8KBench(Benchmark):
    def init_dataset(self):
        raw_datasets = load_dataset("gsm8k", "main")
        self.dataset = [
            dspy.Example(**x).with_inputs("question") for x in raw_datasets["train"]
        ]
        self.test_set = [
            dspy.Example(**x).with_inputs("question") for x in raw_datasets["test"]
        ]
