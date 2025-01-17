import dspy

import langProBe.program as program


class GenerateAnswer(dspy.Signature):
    """Answer multiple choice questions."""

    question = dspy.InputField()
    answer = dspy.OutputField()


HotPotQAPredict = program.Predict(GenerateAnswer)
HotPotQACoT = program.CoT(GenerateAnswer)
HotPotQARAG = program.RAG(GenerateAnswer)
HotPotQASimplifiedBaleen = program.SimplifiedBaleen(GenerateAnswer)
HotPotQAGeneratorCriticRanker = program.GeneratorCriticRanker(GenerateAnswer)
HotPotQAGeneratorCriticFuser = program.GeneratorCriticFuser(GenerateAnswer)
