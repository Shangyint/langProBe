from langProBe.benchmark import Benchmark
from datasets import load_dataset
import dspy


class Overthinking(Benchmark):
    def init_dataset(self):
        raw_dataset = load_dataset("AlexCuadron/best_of_o1")
