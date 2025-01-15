import copy
from ..benchmark import Benchmark
import dspy
from datasets import load_dataset


class IrisTypoBench(Benchmark):
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

        self.test_set = self.dataset[len(self.dataset) // 2 :]
        self.dataset = self.dataset[: len(self.dataset) // 2]
        self.train_set = self.dataset[:15]
        self.val_set = self.dataset[15:]
        self.dev_set = self.dataset
