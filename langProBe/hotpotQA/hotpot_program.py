import dspy
import itertools
import re
from .hotpot_utils import deduplicate, decide_model_type
from dspy.retrieve.websearch import BraveSearch





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

class GenerateSearchQueryList(dspy.Signature):
    """Given the question and its four options, extract the relevant keywords, phrases, or facts needed to perform an effective web search. Return three most important keywords/phrases or facts in a comma-separated format."""

    question = dspy.InputField()
    query = dspy.OutputField(desc="Please be short and concise.")


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


class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        colbertv2_wiki17_abstracts = dspy.ColBERTv2(
            url="http://20.102.90.50:2017/wiki17_abstracts"
        )
        dspy.settings.configure(rm=colbertv2_wiki17_abstracts)

        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer_with_context)
    
    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)
    