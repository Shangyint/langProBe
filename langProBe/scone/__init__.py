from .scone_data import SconeBench
from .scone_program import *
from langProBe.benchmark import BenchmarkMeta
import dspy


benchmark = [
    BenchmarkMeta(
        SconeBench,
        [SconePredict, SconeCoT, SconeGeneratorCriticRanker, SconeGeneratorCriticFuser],
        dspy.evaluate.answer_exact_match,
    )
]
