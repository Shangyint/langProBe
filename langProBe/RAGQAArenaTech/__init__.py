import dspy.evaluate
from .RAGQAArenaTech_data import RAGQAArenaBench
from .RAGQAArenaTech_program import RAGQACoT, RAGQAPredict, RAGQARAG, RAGQASimplifiedBaleen
from langProBe.benchmark import BenchmarkMeta
import dspy

benchmark = [
    BenchmarkMeta(
        RAGQAArenaBench, [RAGQACoT, RAGQAPredict, RAGQARAG, RAGQASimplifiedBaleen], dspy.evaluate.SemanticF1()
    )
]