from langProBe.benchmark import Benchmark
from datasets import load_dataset
import dspy


class JudgeBench(Benchmark):
    def init_dataset(self):
        raw_dataset = load_dataset("ScalerLab/JudgeBench")
        self.dataset = [
            dspy.Example(**x).with_inputs("question", "response_A", "response_B")
            for x in raw_dataset["claude"]
        ]
        self.test_set = [
            dspy.Example(**x).with_inputs("question", "response_A", "response_B")
            for x in raw_dataset["gpt"]
        ]
