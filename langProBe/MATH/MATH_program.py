import dspy
import langProBe.program as program


class GenerateAnswerBasic(dspy.Signature):
    """
    Solve the problem step by step. List your reasoning for each step.
    """

    problem = dspy.InputField(format=str)
    answer = dspy.OutputField(desc="The answer to the problem only, no text or explanations.")


MATHCoT = program.CoT(GenerateAnswerBasic)
MATHGeneratorCriticFuser = program.GeneratorCriticFuser(GenerateAnswerBasic)
MATHGeneratorCriticRanker = program.GeneratorCriticRanker(GenerateAnswerBasic)
