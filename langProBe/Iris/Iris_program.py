import dspy
import langProBe.program as program


class Sig(dspy.Signature):
    "Given the petal and sepal dimensions in cm, predict the iris species."

    petal_length = dspy.InputField()
    petal_width = dspy.InputField()
    sepal_length = dspy.InputField()
    sepal_width = dspy.InputField()
    answer = dspy.OutputField(desc="setosa, versicolor, or virginica")

class Sig_typo(dspy.Signature):
    "Given the petal and sepal dimensions in cm, predict the iris species."

    petal_length = dspy.InputField()
    petal_width = dspy.InputField()
    sepal_length = dspy.InputField()
    sepal_width = dspy.InputField()
    answer = dspy.OutputField(desc="setosa, versicolour, or virginica")

IrisPredict = program.Predict(Sig)
IrisCot = program.CoT(Sig)
IrisGeneratorCriticRanker = program.GeneratorCriticRanker(Sig)
IrisGeneratorCriticFuser = program.GeneratorCriticFuser(Sig)

IrisTypoPredict = program.Predict(Sig_typo)
IrisTypoCot = program.CoT(Sig_typo)
IrisTypoGeneratorCriticRanker = program.GeneratorCriticRanker(Sig_typo)
IrisTypoGeneratorCriticFuser = program.GeneratorCriticFuser(Sig_typo)
