from .Iris_data import IrisBench
from .Iris_program import (
    IrisPredict,
    IrisCot,
    IrisGeneratorCriticFuser,
    IrisGeneratorCriticRanker,
)
from langProBe.benchmark import BenchmarkMeta
import dspy


# benchmark = [
#     BenchmarkMeta(
#         IrisBench,
#         [IrisPredict, IrisCot, IrisGeneratorCriticFuser, IrisGeneratorCriticRanker],
#         dspy.evaluate.answer_exact_match,
#     )
# ]
benchmark = [
    BenchmarkMeta(
        IrisBench,
        [IrisPredict],
        dspy.evaluate.answer_exact_match,
    )
]