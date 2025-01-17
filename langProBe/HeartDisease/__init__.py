from .HeartDisease_data import HeartDiseaseBench
from .HeartDisease_program import (
    HeartDiseaseClassify,
    HeartDiseasePredict,
    HeartDiseaseCoT,
)
from langProBe.benchmark import BenchmarkMeta
import dspy

programs = [HeartDiseasePredict, HeartDiseaseCoT, HeartDiseaseClassify()]

benchmark = [BenchmarkMeta(HeartDiseaseBench, programs, lambda g,p,t=None: g.answer == p.answer)]

