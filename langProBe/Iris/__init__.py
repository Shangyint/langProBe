from .Iris_data import IrisBench
from .Iris_program import IrisCot, IrisGeneratorCriticFuser, IrisGeneratorCriticRanker
from langProBe.benchmark import BenchmarkMeta
import dspy


benchmark = [
    BenchmarkMeta(
        IrisBench,
        [IrisCot, IrisGeneratorCriticFuser, IrisGeneratorCriticRanker],
        dspy.evaluate.answer_exact_match,
    )
]
