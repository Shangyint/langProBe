from ..benchmark import Benchmark
import dspy
import ujson
import urllib.request
from pathlib import Path


class RAGQAArenaBench(Benchmark):
    def init_dataset(self):
        Path("langProBe/RAGQAArenaTech/data").mkdir(exist_ok=True)

        urllib.request.urlretrieve(
            "https://huggingface.co/dspy/cache/resolve/main/ragqa_arena_tech_500.json",
            "./langProBe/RAGQAArenaTech/data/ragqa_arena_tech_500.json",
        )

        with open("langProBe/RAGQAArenaTech/data/ragqa_arena_tech_500.json") as f:
            raw_datasets = ujson.load(f)
        self.dataset = [
            dspy.Example(**x).with_inputs(
                "question",
            )
            for x in raw_datasets
        ]

        self.test_set = self.dataset[len(self.dataset) // 2 :]
        self.dataset = self.dataset[: len(self.dataset) // 2]
        self.train_set = self.dataset[:50]
        self.val_set = self.dataset[50:]
        self.dev_set = self.dataset