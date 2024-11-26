import dspy
import langProBe.program as program


class ScoNeSignature(dspy.Signature):
    ("""context, question -> answer""")

    context = dspy.InputField()
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Yes or No")


SconePredict = program.Predict(ScoNeSignature)
SconeCoT = program.CoT(ScoNeSignature)
SconeGeneratorCriticRanker = program.GeneratorCriticRanker(ScoNeSignature)
SconeGeneratorCriticFuser = program.GeneratorCriticFuser(ScoNeSignature)
