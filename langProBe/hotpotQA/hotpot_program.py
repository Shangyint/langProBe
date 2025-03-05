import dspy

import langProBe.dspy_program as dspy_program


class GenerateAnswer(dspy.Signature):
    """Answer multiple choice questions."""

    question = dspy.InputField()
    answer = dspy.OutputField()


HotPotQAPredict = dspy_program.Predict(GenerateAnswer)
HotPotQACoT = dspy_program.CoT(GenerateAnswer)
HotPotQARAG = dspy_program.RAG(GenerateAnswer)
HotPotQASimplifiedBaleen = dspy_program.SimplifiedBaleen(GenerateAnswer)
HotPotQAGeneratorCriticRanker = dspy_program.GeneratorCriticRanker(GenerateAnswer)
HotPotQAGeneratorCriticFuser = dspy_program.GeneratorCriticFuser(GenerateAnswer)

HotPotQAGeneratorCriticRanker_20 = dspy_program.GeneratorCriticRanker(
    GenerateAnswer, n=20
)
HotPotQAGeneratorCriticFuser_20 = dspy_program.GeneratorCriticFuser(
    GenerateAnswer, n=20
)
