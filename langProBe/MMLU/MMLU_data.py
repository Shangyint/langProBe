from ..benchmark import Benchmark
import dspy
from datasets import load_dataset


class MMLUBench(Benchmark):
    def init_dataset(self):
        self.raw_datasets = load_dataset("cais/mmlu", "all")

        option_map = {0: "A", 1: "B", 2: "C", 3: "D"}
        self.dataset = []

        for x in self.raw_datasets["test"]:
            x["question"] = "\n".join(
                [
                    x["question"],
                    "A. " + x["choices"][0],
                    "B. " + x["choices"][1],
                    "C. " + x["choices"][2],
                    "D. " + x["choices"][3],
                ]
            )
            x["answer"] = option_map[x["answer"]]
            self.dataset.append(dspy.Example(**x).with_inputs("question"))
