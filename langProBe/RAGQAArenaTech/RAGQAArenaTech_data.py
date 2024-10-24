from ..benchmark import Benchmark
import dspy
import ujson
import urllib.request
from pathlib import Path


class RAGQAArenaBench(Benchmark):
    def init_dataset(self):
        Path("data").mkdir(exist_ok=True)

        urllib.request.urlretrieve(
            "https://huggingface.co/dspy/cache/resolve/main/ragqa_arena_tech_500.json",
            "data/ragqa_arena_tech_500.json",
        )
        with open("data/ragqa_arena_tech_500.json") as f:
            raw_datasets = ujson.load(f)
        self.dataset = [
            dspy.Example(**x).with_inputs(
                "question",
            )
            for x in raw_datasets
        ]
