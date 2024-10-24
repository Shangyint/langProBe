import dspy
from .MMLU_utils import deduplicate


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


class GenerateSearchQuery(dspy.Signature):
    """Write a simple search query that will help answer a complex question."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField()


class CoT(dspy.Module):
    def __init__(self):
        self.prog = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        pred = self.prog(question=question)
        return pred


class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer_with_context)

    def forward(self, question):
        context = self.retrieve(question).passages
        pred = self.generate_answer(context=context, question=question)
        return pred


class SimplifiedBaleen(dspy.Module):
    def __init__(self, passages_per_hop=2, max_hops=2):
        super().__init__()
        self.generate_query = [
            dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)
        ]
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.max_hops = max_hops

    def forward(self, question):
        context = []
        prev_queries = [question]

        for hop in range(self.max_hops):
            query = self.generate_query[hop](context=context, question=question).query
            prev_queries.append(query)
            passages = self.retrieve(query).passages
            context = deduplicate(context + passages)

        pred = self.generate_answer(context=context, question=question)
        return pred
