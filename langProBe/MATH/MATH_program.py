import dspy
import langProBe.program as program


class GenerateAnswerBasic(dspy.Signature):
    """
    Solve the problem step by step. List your reasoning for each step.
    """

    question = dspy.InputField(format=str)
    answer = dspy.OutputField(desc="The answer to the problem")


MATHCoT = program.CoT(GenerateAnswerBasic)
MATHGeneratorCriticFuser = program.GeneratorCriticFuser(GenerateAnswerBasic)
MATHGeneratorCriticRanker = program.GeneratorCriticRanker(GenerateAnswerBasic)
