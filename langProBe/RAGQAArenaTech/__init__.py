import dspy.evaluate
from .RAGQAArenaTech_data import RAGQAArenaBench
from .RAGQAArenaTech_program import *
from langProBe.benchmark import BenchmarkMeta
import dspy

eval_lm = dspy.LM("openai/gpt-4o")
eval_module = dspy.evaluate.SemanticF1()
eval_module.set_lm(eval_lm)

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
        eval_module,
    )
]
