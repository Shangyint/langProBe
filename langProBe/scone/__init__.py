from .scone_data import SconeBench
from .scone_program import (
    SconeCoT,
    SconeGeneratorCriticRanker,
    SconeGeneratorCriticFuser,
)
from langProBe.benchmark import BenchmarkMeta
import dspy


benchmark = [
    BenchmarkMeta(
        SconeBench,
        [SconeCoT, SconeGeneratorCriticRanker, SconeGeneratorCriticFuser],
        dspy.evaluate.answer_exact_match,
    )
]
