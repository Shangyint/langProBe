from .HeartDisease_data import HeartDiseaseBench
from .HeartDisease_program import Classify
from langProBe.benchmark import BenchmarkMeta
import dspy

programs = [Classify()]

benchmark = [BenchmarkMeta(HeartDiseaseBench, programs, dspy.evaluate.answer_exact_match)]

