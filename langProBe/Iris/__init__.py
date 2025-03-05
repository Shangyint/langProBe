from .Iris_data import IrisBench
from .Iris_program import *
from langProBe.benchmark import BenchmarkMeta
import dspy


benchmark = [
    BenchmarkMeta(
        IrisBench,
        [
            IrisPredict,
            IrisCot,
            IrisGeneratorCriticFuser,
            IrisGeneratorCriticRanker,
            IrisGeneratorCriticFuser_20,
            IrisGeneratorCriticRanker_20,
        ],
        dspy.evaluate.answer_exact_match,
    )
]
