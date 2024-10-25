from .scone_data import SconeBench
from .scone_program import CoT
from langProBe.benchmark import BenchmarkMeta
import dspy

programs = [CoT]

benchmark = [BenchmarkMeta(SconeBench, programs, dspy.evaluate.answer_exact_match)]
