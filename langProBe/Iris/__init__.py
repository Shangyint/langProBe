from .Iris_data import IrisBench
from .Iris_program import (
    IrisPredict,
    IrisCot,
    IrisGeneratorCriticFuser,
    IrisGeneratorCriticRanker,
    IrisTypoPredict,
    IrisTypoCot,
    IrisTypoGeneratorCriticFuser,
    IrisTypoGeneratorCriticRanker
    
)
from langProBe.benchmark import BenchmarkMeta
import dspy


benchmark = [
    BenchmarkMeta(
        IrisBench,
        [IrisPredict, IrisCot, IrisGeneratorCriticFuser, IrisGeneratorCriticRanker],
        dspy.evaluate.answer_exact_match,
    ),
    BenchmarkMeta(
        IrisBench,
        [IrisTypoPredict, IrisTypoCot, IrisTypoGeneratorCriticFuser, IrisTypoGeneratorCriticRanker],
        dspy.evaluate.answer_exact_match
    )
]
