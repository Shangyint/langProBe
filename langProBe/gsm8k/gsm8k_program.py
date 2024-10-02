import dspy


class NaiveProgram(dspy.Module):
    def __init__(self):
        self.prog = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        pred = self.prog(question=question)
        return pred
    

class SelfCriticGenerator(dspy.Signature):
    """
    Solve the math problem step by step. List your reasoning for each step.
    """
    question = dspy.InputField(format=str)
    answer = dspy.OutputField(desc="The answer to the math problem")

class SelfCriticCritic(dspy.Signature):
    """
    Provide feedback on each reasoning step.
    """
    reasoning = dspy.InputField(format=str)
    feedback = dspy.OutputField(desc="Feedback on the reasoning step")

class SelfCritic(dspy.Module):
    def __init__(self):
        self.prog = dspy.ChainOfThought(SelfCriticGenerator)