import dspy
import langProBe.dspy_program as dspy_program


class GenerateAnswerBasic(dspy.Signature):
    """
    Solve the problem step by step. List your reasoning for each step.
    """

    problem = dspy.InputField(format=str)
    answer = dspy.OutputField(
        desc="The answer to the problem only, no text or explanations."
    )


MATHPredict = dspy_program.Predict(GenerateAnswerBasic)
MATHCoT = dspy_program.CoT(GenerateAnswerBasic)
MATHGeneratorCriticFuser = dspy_program.GeneratorCriticFuser(GenerateAnswerBasic)
MATHGeneratorCriticFuser_20 = dspy_program.GeneratorCriticFuser(
    GenerateAnswerBasic, n=20
)
MATHGeneratorCriticRanker = dspy_program.GeneratorCriticRanker(GenerateAnswerBasic)
MATHGeneratorCriticRanker_20 = dspy_program.GeneratorCriticRanker(
    GenerateAnswerBasic, n=20
)
