from ..benchmark import Benchmark
import dspy
from datasets import load_dataset
import random


class HeartDiseaseBench(Benchmark):
    mappings = {
        "sex": {0: "female", 1: "male"},
        "cp": {
            1: "typical angina",
            2: "atypical angina",
            3: "non-anginal pain",
            4: "asymptomatic",
        },
        "restecg": {
            0: "normal",
            1: "ST-T wave abnormality",
            2: "left ventricular hypertrophy",
        },
        "exang": {0: "no", 1: "yes"},
        "slope": {1: "upsloping", 2: "flat", 3: "downsloping"},
        "thal": {"3": "normal", "6": "fixed defect", "7": "reversible defect"},
        "target": {0: "no", 1: "yes"},  # presence of heart disease
    }

    def init_dataset(self):
        dataset = load_dataset("buio/heart-disease")
        fullset = []

        for x in dataset["train"]:
            for key, value in x.items():
                if key in self.mappings:
                    x[key] = self.mappings[key].get(value, value)
                x[key] = str(x[key])

            inputs = [k for k in x.keys() if k != "target"]
            x["answer"] = x["target"]
            del x["target"]

            fullset.append(dspy.Example(**x).with_inputs(*inputs))

        random.Random(0).shuffle(fullset)
        self.dataset = fullset
