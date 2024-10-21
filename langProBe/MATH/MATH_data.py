from ..benchmark import Benchmark
import dspy

from datasets import load_dataset
import random


class MATHBench(Benchmark):
    def init_dataset(self):
        raw_datasets = load_dataset("lighteval/MATH", "all")["test"]
        self.dataset = [dspy.Example(**x).with_inputs("problem") for x in raw_datasets]
