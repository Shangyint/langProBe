from .Iris_data import IrisBench
from .Iris_program import Classify
from langProBe.benchmark import BenchmarkMeta
import dspy


benchmark = [BenchmarkMeta(IrisBench, [Classify], dspy.evaluate.answer_exact_match)]
