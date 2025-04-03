from ..benchmark import Benchmark
import dspy
from datasets import load_dataset

class MedQABench(Benchmark):
    def init_dataset(self):
        raw_datasets = load_dataset("bigbio/med_qa", name="med_qa_en_4options_source")
        raw_datasets = raw_datasets.map(self.format_data)
        
        self.dataset = [
            dspy.Example(**x).with_inputs("question", "options") for x in raw_datasets["train"]
        ]

        self.val_set = [
            dspy.Example(**x).with_inputs("question", "options") for x in raw_datasets["validation"]
        ]
        
        self.test_set = [
            dspy.Example(**x).with_inputs("question", "options") for x in raw_datasets["test"]
        ]
    
    def format_data(self, example):
        example["options"] = [item["value"] for item in example["options"]]
        return example