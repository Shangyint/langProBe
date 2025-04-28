from langProBe.overthinking.overthinking_program import (
    OverthinkingGeneratorCriticFuser_20,
    OverthinkingGeneratorCriticRanker_20,
    OverthinkingCoT,
    OverthinkingPredict,
    OverthinkingGeneratorCriticFuser,
    OverthinkingGeneratorCriticRanker,
)
from .overthinking_data import Overthinking
from .overthinking_program import *

from langProBe.benchmark import BenchmarkMeta


def llm_overthinking_eval(gold, pred, target: str = None):
    pass


benchmark = [
    BenchmarkMeta(
        Overthinking,
        [
            OverthinkingPredict,
            OverthinkingCoT,
            OverthinkingGeneratorCriticFuser,
            OverthinkingGeneratorCriticRanker,
            OverthinkingGeneratorCriticFuser_20,
            OverthinkingGeneratorCriticRanker_20,
        ],
        llm_overthinking_eval,
    )
]
