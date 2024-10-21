from typing import Any, Callable
from .irera_data import IReRaBench
from .irera_program import Infer, InferRetrieve, InferRetrieveRank
from langProBe.benchmark import Benchmark
import dspy


benchmark: Callable[[], Benchmark] = IReRaBench
programs = [Infer, InferRetrieve, InferRetrieveRank]
metric = dspy.evaluate.answer_exact_match
