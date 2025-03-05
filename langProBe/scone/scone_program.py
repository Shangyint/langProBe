import dspy
import langProBe.dspy_program as dspy_program


class ScoNeSignature(dspy.Signature):
    ("""context, question -> answer""")

    context = dspy.InputField()
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Yes or No")


SconePredict = dspy_program.Predict(ScoNeSignature)
SconeCoT = dspy_program.CoT(ScoNeSignature)
SconeGeneratorCriticRanker = dspy_program.GeneratorCriticRanker(ScoNeSignature)
SconeGeneratorCriticRanker_20 = dspy_program.GeneratorCriticRanker(ScoNeSignature, n=20)
SconeGeneratorCriticFuser = dspy_program.GeneratorCriticFuser(ScoNeSignature)
SconeGeneratorCriticFuser_20 = dspy_program.GeneratorCriticFuser(ScoNeSignature, n=20)
