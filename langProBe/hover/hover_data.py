from ..benchmark import Benchmark
import dspy
from datasets import load_dataset
import tqdm
import random
from .hover_utils import count_unique_docs


class hoverBench(Benchmark):
    def init_dataset(self):
        dataset = load_dataset("hover")

        hf_trainset = dataset["train"]
        hf_testset = dataset["validation"]     # Using validation dataset because test dataset is not labeled

        reformatted_hf_trainset = []
        reformatted_hf_testset = []

        for example in tqdm.tqdm(hf_trainset):
            claim = example["claim"]
            supporting_facts = example["supporting_facts"]
            label = example["label"]

            if count_unique_docs(example) == 3:  # Limit to 3 hop examples
                reformatted_hf_trainset.append(
                    dict(claim=claim, supporting_facts=supporting_facts, label=label)
                )

        for example in tqdm.tqdm(hf_testset):
            claim = example["claim"]
            supporting_facts = example["supporting_facts"]
            label = example["label"]

            reformatted_hf_testset.append(
                dict(claim=claim, supporting_facts=supporting_facts, label=label)
            )

        rng = random.Random()
        rng.seed(0)
        rng.shuffle(reformatted_hf_trainset)
        rng = random.Random()
        rng.seed(1)
        rng.shuffle(reformatted_hf_testset)

        trainset = reformatted_hf_trainset
        testset = reformatted_hf_testset

        trainset = [dspy.Example(**x).with_inputs("claim") for x in trainset]
        testset = [dspy.Example(**x).with_inputs("claim") for x in testset]

        self.dataset = trainset
        self.test_set = testset