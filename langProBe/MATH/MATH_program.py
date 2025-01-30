import dspy
import langProBe.program as program


class GenerateAnswerBasic(dspy.Signature):
    """
    Solve the problem step by step. List your reasoning for each step.
    """

    problem = dspy.InputField(format=str)
    answer = dspy.OutputField(
        desc="The answer to the problem only, no text or explanations."
    )


MATHPredict = program.Predict(GenerateAnswerBasic)
MATHCoT = program.CoT(GenerateAnswerBasic)
MATHGeneratorCriticFuser = program.GeneratorCriticFuser(GenerateAnswerBasic)
MATHGeneratorCriticFuser_20 = program.GeneratorCriticFuser(GenerateAnswerBasic, n=20)
MATHGeneratorCriticRanker = program.GeneratorCriticRanker(GenerateAnswerBasic)
MATHGeneratorCriticRanker_20 = program.GeneratorCriticRanker(GenerateAnswerBasic, n=20)
