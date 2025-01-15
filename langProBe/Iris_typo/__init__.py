from .Iris_typo_data import IrisTypoBench
from .Iris_typo_program import (
    IrisTypoPredict,
    IrisTypoCot,
    IrisTypoGeneratorCriticFuser,
    IrisTypoGeneratorCriticRanker,
)
from langProBe.benchmark import BenchmarkMeta
import dspy


benchmark = [
    BenchmarkMeta(
        IrisTypoBench,
        [IrisTypoPredict, IrisTypoCot, IrisTypoGeneratorCriticFuser, IrisTypoGeneratorCriticRanker],
        dspy.evaluate.answer_exact_match,
    )
]
