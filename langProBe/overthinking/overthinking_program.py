import dspy
import langProBe.dspy_program as dspy_program


class LLMOverthinkingSignature(dspy.Signature):
    """
    Compare two responses to a question, and determine which is better.
    """

    question = dspy.InputField()

    answer = dspy.OutputField(desc="Is this model overthinking?")


OverthinkingPredict = dspy_program.Predict(LLMOverthinkingSignature)
OverthinkingCoT = dspy_program.CoT(LLMOverthinkingSignature)
OverthinkingGeneratorCriticFuser = dspy_program.GeneratorCriticFuser(LLMOverthinkingSignature)
OverthinkingGeneratorCriticFuser_20 = dspy_program.GeneratorCriticFuser(
    LLMOverthinkingSignature, n=20
)
OverthinkingGeneratorCriticRanker_20 = dspy_program.GeneratorCriticRanker(
    LLMOverthinkingSignature, n=20
)
