import dspy


class GenerateAnswer_with_context(dspy.Signature):
    """Answer multiple choice questions."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(
        desc="Do not write explanations or additional text. Just select the answer."
    )

class GenerateAnswer(dspy.Signature):
    """Answer multiple choice questions."""

    question = dspy.InputField()
    answer = dspy.OutputField(
        desc="Do not write explanations or additional text. Just select the answer."
    )


class CoT_with_context(dspy.Module):
    def __init__(self):
        self.prog = dspy.ChainOfThought(GenerateAnswer_with_context)

    def forward(self, question):
        context = None  #TODO: how should we get the context
        pred = self.prog(context=context, question=question)
        return pred

class CoT(dspy.Module):
    def __init__(self):
        self.prog = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        pred = self.prog(question=question)
        return pred
