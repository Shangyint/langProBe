from ..benchmark import Benchmark
import dspy
from datasets import load_dataset
import random


class IrisBench(Benchmark):
    def init_dataset(self):
        raw_dataset = load_dataset("hitorilabs/iris")

        self.dataset = [                         
            dspy.Example(**{k: str(round(v, 2)) for k, v in example.items()})
            for example in raw_dataset["train"]
        ]
        self.dataset = [
            dspy.Example(
                **{
                    **x,
                    "answer": ["setosa", "versicolor", "virginica"][int(x["species"])],
                }
            )
            for x in self.dataset
        ]
        self.dataset = [
            x.with_inputs("petal_length", "petal_width", "sepal_length", "sepal_width")
            for x in self.dataset
        ]
        random.Random(0).shuffle(self.dataset)
        