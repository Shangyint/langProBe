from ..benchmark import Benchmark
import dspy
from datasets import load_dataset

class PubMedQABench(Benchmark):
    def init_dataset(self):
        raw_datasets = load_dataset("qiaojin/PubMedQA", "pqa_artificial")
        raw_datasets_test = load_dataset("qiaojin/PubMedQA", "pqa_labeled")

        raw_datasets = raw_datasets.map(lambda x: {"answer": x["final_decision"], **x})
        raw_datasets_test = raw_datasets_test.map(lambda x: {"answer": x["final_decision"], **x})
        
        self.dataset = [
            dspy.Example(**x).with_inputs("question") for x in raw_datasets["train"]
        ]

        self.test_set = [
            dspy.Example(**x).with_inputs("question") for x in raw_datasets_test["train"]
        ]