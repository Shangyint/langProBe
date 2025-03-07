from ..benchmark import Benchmark
import dspy
from datasets import load_dataset

class MedMCQABench(Benchmark):
    def init_dataset(self):
        raw_datasets = load_dataset("openlifescienceai/medmcqa")
        raw_datasets = raw_datasets.map(self.format_data)
        
        self.dataset = [
            dspy.Example(**x).with_inputs("question", "options") for x in raw_datasets["train"]
        ]

        self.test_set = [
            dspy.Example(**x).with_inputs("question", "options") for x in raw_datasets["validation"]
        ]
    
    def format_data(self, example):
        example["options"] = [example["opa"], example["opb"], example["opc"], example["opd"]]
        example["answer"] = example["options"][example["cop"]]
        return example