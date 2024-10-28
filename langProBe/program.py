import dspy
from copy import deepcopy
from dsp.utils import deduplicate


def CoTAdapter(signature):
    return dspy.ChainOfThought(signature)


def default_input_to_query(**kwargs):
    if len(kwargs) == 1:
        return list(kwargs.values())[0]
    else:
        raise ValueError("Cannot convert multiple inputs to a query, please specify input_to_query.")

def RAGAdapter(signature, retriever=dspy.Retrieve(k=3), input_to_query=default_input_to_query):
    class RAG(dspy.Module):
        def __init__(self):
            self.retriver = retriever
            self.prog = dspy.ChainOfThought(signature)
            self.prog._predict.signature = self.prog._predict.signature.append(
                "context", dspy.InputField(desc="may contain relevant facts")
            )
            self.prog._predict.extended_signature = self.prog._predict.signature

        def forward(self, **kwargs):
            context = self.retriver(input_to_query(**kwargs)).passages
            pred = self.prog(context=context, **kwargs)
            return pred

    return RAG()


def SimplifiedBaleenAdapter(signature, query_gen_input=None, retriever=dspy.Retrieve(k=2), max_hops=2):
    """
    args:
        signature: The signature to the final generate module
        query_gen_input: a list of keywords to be used as input to the query generation module
        retriever: a retriever module to be used to retrieve relevant facts
        max_hops: the number of hops to be used in the simplified
        FIXME (shangyin) correctly handle query_gen_input
    """
    class SimplifiedBaleen(dspy.Module):
        def __init__(self):
            self.max_hops = max_hops
            self.retriever = retriever
            verified_signature = dspy.ensure_signature(signature)
            verified_signature = verified_signature.append("context", dspy.InputField(desc="may contain relevant facts"))

            # remove the output field from the generate query signature
            # generate query should use a default instruction rather than instruction for the original signature
            # FIXME (shangyin) fix the default signature.instructions
            input_fields = verified_signature.input_fields
            generate_query_signature = dspy.Signature(input_fields)
            generate_query_signature = generate_query_signature.append("search_query", dspy.OutputField())


            self.generate_query = [
                dspy.ChainOfThought(generate_query_signature) for _ in range(self.max_hops)
            ]
            self.generate_answer = dspy.ChainOfThought(verified_signature)

        def forward(self, **kwargs):
            context = []

            for hop in range(self.max_hops):
                query = self.generate_query[hop](context=context, **kwargs).search_query
                passages = self.retriever(query).passages
                context = deduplicate(context + passages)

            pred = self.generate_answer(context=context, **kwargs)
            return pred

    return SimplifiedBaleen()


if __name__ == "__main__":
    # Example usage
    dspy.configure(
        lm=dspy.LM("openai/gpt-4o-mini"),
        rm=dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts"),
    )

    question = "What is the capital of France?"
    context = "France is a country in Europe."

    # CoT
    print("======== CoT =========")
    cot = CoTAdapter("question, context -> answer")
    cot(question=question, context=context)
    dspy.settings.lm.inspect_history()

    # RAG
    print("======== RAG =========")
    rag = RAGAdapter("question -> answer")
    rag(question=question)
    dspy.settings.lm.inspect_history()

    # SimplifiedBaleen
    print("======== SimplifiedBaleen =========")
    simplified_baleen = SimplifiedBaleenAdapter("question -> answer")
    simplified_baleen(question=question)
    dspy.settings.lm.inspect_history()
