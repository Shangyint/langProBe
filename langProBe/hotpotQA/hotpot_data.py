from ..benchmark import Benchmark
import dspy
from datasets import load_dataset


class HotpotQABench(Benchmark):
    def init_dataset(self):
        raw_datasets = load_dataset("hotpot_qa", "distractor")
        self.dataset = [dspy.Example(**x).with_inputs("question", "context") for x in raw_datasets['train']]

    def create_splits(self):
        total_len = len(self.dataset)
        self.test_set = self.dataset[:int(0.8 * total_len)]
        self.dev_set = self.dataset[int(0.8 * total_len):int(0.9 * total_len)]
        self.train_set = self.dataset[int(0.9 * total_len):]

