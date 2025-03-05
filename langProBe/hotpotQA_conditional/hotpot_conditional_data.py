from ..benchmark import Benchmark
import pandas as pd
import os
import dspy


class HotpotQAConditionalBench(Benchmark):
    def init_dataset(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))

        hotpotqa_train_path = os.path.join(script_dir, "./data/hotpot_train.csv")
        hotpotqa_dev_path = os.path.join(script_dir, "./data/hotpot_dev.csv")
        hotpotqa_test_path = os.path.join(script_dir, "./data/hotpot_test.csv")

        # Read the datasets
        hotpotqa_train = pd.read_csv(hotpotqa_train_path)
        hotpotqa_dev = pd.read_csv(hotpotqa_dev_path)
        hotpotqa_test = pd.read_csv(hotpotqa_test_path)

        # Load and configure the datasets.
        trainset = [
            dspy.Example(
                question=row["question"],
                answer=row["answer"],
                category=row["answer category"],
            ).with_inputs("question")
            for index, row in hotpotqa_train.iterrows()
        ]
        testset = [
            dspy.Example(
                question=row["question"],
                answer=row["answer"],
                category=row["answer category"],
            ).with_inputs("question")
            for index, row in hotpotqa_test.iterrows()
        ]
        devset = [
            dspy.Example(
                question=row["question"],
                answer=row["answer"],
                category=row["answer category"],
            ).with_inputs("question")
            for index, row in hotpotqa_dev.iterrows()
        ]

        self.dataset = trainset + testset + devset
        self.train_set = trainset
        self.dev_set = devset
        self.val_set = devset
        self.test_set = testset
