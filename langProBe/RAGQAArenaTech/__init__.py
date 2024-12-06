import dspy.evaluate
from .RAGQAArenaTech_data import RAGQAArenaBench
from .RAGQAArenaTech_program import (
    RAGQACoT,
    RAGQAPredict,
    RAGQARAG,
    RAGQASimplifiedBaleen,
    RAGQAGeneratorCriticFuser,
    RAGQAGeneratorCriticRanker,
)
from langProBe.benchmark import BenchmarkMeta
import dspy

benchmark = [
    BenchmarkMeta(
        RAGQAArenaBench,
        [
            RAGQASimplifiedBaleen,
            RAGQACoT,
            RAGQAPredict,
            RAGQARAG,
            RAGQAGeneratorCriticRanker,
            RAGQAGeneratorCriticFuser,
        ],
        dspy.evaluate.SemanticF1(),
    )
]
