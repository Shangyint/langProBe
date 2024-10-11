from ..benchmark import Benchmark
import dspy

from datasets import load_dataset
import random


class MATHBench(Benchmark):
    def init_dataset(self):
        raw_datasets = load_dataset("lighteval/MATH-Hard", "default")["test"]
        self.dataset = [dspy.Example(**x).with_inputs("problem") for x in raw_datasets]

    def create_splits(self):
        random.seed(0)
        random.shuffle(self.dataset)

        self.train_set, self.dev_set, self.test_set = (
            self.dataset[:100],
            self.dataset[100:200],
            self.dataset[200:500],
        )
