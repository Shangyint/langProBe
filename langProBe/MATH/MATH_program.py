import dspy


class CoT(dspy.Module):
    def __init__(self):
        self.prog = dspy.ChainOfThought("problem -> answer")

    def forward(self, problem):
        pred = self.prog(problem=problem)
        return pred


class GenerateAnswerBasic(dspy.Signature):
    """
    Solve the problem step by step. List your reasoning for each step.
    """

    question = dspy.InputField(format=str)
    answer = dspy.OutputField(desc="The answer to the problem")


class SelfCriticCritic(dspy.Signature):
    """
    Provide feedback and critics on each reasoning step.
    """

    question = dspy.InputField(format=str)
    prev_reasoning = dspy.InputField(
        desc="Your Previous reasoning to solve the math problem"
    )
    feedback = dspy.OutputField(desc="Feedback on the reasoning step")


class GenerateAnswer(dspy.Signature):
    """
    Generate the answer to the problem based on previous reasoning and feedback.
    """

    question = dspy.InputField(desc="The problem to solve")
    prev_reasoning = dspy.InputField(desc="The reasoning to solve the problem")
    feedback = dspy.InputField(desc="Feedback on the reasoning step")
    answer = dspy.OutputField(
        desc="Based on previous reasoning and feedback, answer to the problem. Answer only, no explanation."
    )


class SelfCritic(dspy.Module):
    def __init__(self):
        self.generate_reasoning = dspy.ChainOfThought(GenerateAnswerBasic)
        self.critic_reasoning = dspy.ChainOfThought(SelfCriticCritic)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, problem):
        # FIXME(shangyin): rationale might become deprecated in the future. We should use reasoning instead.
        reasoning = self.generate_reasoning(question=problem).rationale
        feedback = self.critic_reasoning(
            question=problem, prev_reasoning=reasoning
        ).feedback
        answer = self.generate_answer(
            question=problem, prev_reasoning=reasoning, feedback=feedback
        )

        return answer


class MultiChain(dspy.Module):
    def __init__(self, num_chain=5):
        self.num_chain = num_chain
        self.reasoning_generator = dspy.ChainOfThought(
            GenerateAnswerBasic, n=self.num_chain
        )
        self.prog = dspy.MultiChainComparison(GenerateAnswerBasic, M=self.num_chain)

    def forward(self, problem):
        completions = self.reasoning_generator(question=problem)
        pred = self.prog(completions.completions, question=problem)
        return pred
