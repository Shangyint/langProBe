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
        
        option_map = {
            0: "A", 
            1: "B",
            2: "C", 
            3: "D"
        }
        self.test_set, self.dev_set, self.train_set = [], [], []

        for x in self.raw_datasets['test']:
            x['question'] = "\n".join([x['question'], "A. " + x['choices'][0], "B. " + x['choices'][1], "C. " + x['choices'][2], "D. " + x['choices'][3]])
            x['answer'] = option_map[x['answer']]
            self.test_set.append(dspy.Example(**x).with_inputs("question"))
        for x in self.raw_datasets['validation']:
            x['question'] = "\n".join([x['question'], "A. " + x['choices'][0], "B. " + x['choices'][1], "C. " + x['choices'][2], "D. " + x['choices'][3]])
            x['answer'] = option_map[x['answer']]
            self.dev_set.append(dspy.Example(**x).with_inputs("question"))
        for x in self.raw_datasets['dev']:
            x['question'] = "\n".join([x['question'], "A. " + x['choices'][0], "B. " + x['choices'][1], "C. " + x['choices'][2], "D. " + x['choices'][3]])
            x['answer'] = option_map[x['answer']]
            self.train_set.append(dspy.Example(**x).with_inputs("question"))
