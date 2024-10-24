import dspy


class CoT(dspy.Module):
    def __init__(self):
        self.prog = dspy.ChainOfThought("question -> response")

    def forward(self, question):
        pred = self.prog(question=question)
        return pred
