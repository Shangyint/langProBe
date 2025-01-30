import dspy.evaluate
from .RAGQAArenaTech_data import RAGQAArenaBench
from .RAGQAArenaTech_program import *
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
            RAGQAGeneratorCriticFuser_20,
            RAGQAGeneratorCriticRanker_20,
        ],
        dspy.evaluate.SemanticF1(),
    )
]
