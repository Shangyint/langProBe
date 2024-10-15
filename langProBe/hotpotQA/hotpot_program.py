import dspy
import itertools
import re
from hotpot_utils import deduplicate, decide_model_type
from dspy.retrieve.websearch import BraveSearch





turbo = dspy.OpenAI(model="gpt-3.5-turbo", max_tokens=500)


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

class SimplifiedBaleenWithBrave_qlist(dspy.Module):
    def __init__(
        self, passages_per_hop=2, max_hops=2, query_model=None, infer_model=None
    ):
        super().__init__()

        self.generate_query = dspy.Predict(GenerateSearchQueryList)
        self.retrieve = BraveSearch()
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.max_hops = max_hops
        self.query_lm = decide_model_type(query_model) if query_model else turbo
        self.infer_lm = decide_model_type(infer_model) if infer_model else turbo

    def keywords_to_list(self, s):
        keywords = s.split(",")
        return [keyword.strip() for keyword in keywords]

    def forward(self, question):
        history = []
        context = []

        with dspy.context(lm=self.query_lm):
            query_s = self.generate_query(question=question).query
            queries = self.keywords_to_list(query_s)
            history.append(self.query_lm.inspect_history(n=1))
        passages_list = [self.retrieve(query, count=3).passages for query in queries]
        passages_processed = [
            [
                re.sub(r"[<>\\/]", " ", passage["snippet"]).replace("strong", "")
                for passage in passages
            ]
            for passages in passages_list
        ]
        passages_combined = list(itertools.chain(*passages_processed))
        context = deduplicate(context + passages_combined)

        with dspy.context(lm=self.infer_lm):
            pred = self.generate_answer(context=context, question=question)
            history.append(self.infer_lm.inspect_history(n=1))
        return pred.answer, history
