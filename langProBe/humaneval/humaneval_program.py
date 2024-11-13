import dspy

from itertools import chain
from .humaneval_utils import post_process_tests, post_process_code
import langProBe.program as program

NUM_SAMPLES = 20
TEMPARATURE_BASE = 0.7
TEMPARATURE_STEP = 0.01
NUM_TESTS = 5

# Code Generator


class CodeProblem(dspy.Signature):
    prompt = dspy.InputField(format=str)
    code = dspy.OutputField()


HumanEvalPredict = program.Predict(CodeProblem)
HumanEvalCoT = program.CoT(CodeProblem)
HumanEvalGeneratorCriticFuser = program.GeneratorCriticFuser(CodeProblem)
HumanEvalGeneratorCriticRanker = program.GeneratorCriticRanker(CodeProblem)
