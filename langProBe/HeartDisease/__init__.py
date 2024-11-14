from .HeartDisease_data import HeartDiseaseBench
from .HeartDisease_program import Classify, heartdiseasePredict
from langProBe.benchmark import BenchmarkMeta
import dspy

# programs = [Classify(), MMLUPredict]
programs = [heartdiseasePredict]

benchmark = [BenchmarkMeta(HeartDiseaseBench, programs, dspy.evaluate.answer_exact_match)]

