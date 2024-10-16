from ..benchmark import Benchmark
import dspy
from datasets import load_dataset


class MMLUBench(Benchmark):
    def init_dataset(self):
        self.raw_datasets = load_dataset("cais/mmlu", "all")

    def create_splits(self):
        test = self.raw_datasets['test']
        dev = self.raw_datasets['validation']
        train = self.raw_datasets['dev']
        self.test_set = [dspy.Example(**x).with_inputs("question") for x in test]
        self.dev_set = [dspy.Example(**x).with_inputs("question") for x in dev]
        self.train_set = [dspy.Example(**x).with_inputs("question") for x in train]
