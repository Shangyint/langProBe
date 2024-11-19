from .HeartDisease_data import HeartDiseaseBench
from .HeartDisease_program import HeartDiseaseClassify, HeartDiseasePredict, HeartDiseaseCoT
from langProBe.benchmark import BenchmarkMeta
import dspy

programs = [HeartDiseasePredict, HeartDiseaseCoT, HeartDiseaseClassify()]

benchmark = [BenchmarkMeta(HeartDiseaseBench, programs, dspy.evaluate.answer_exact_match)]

