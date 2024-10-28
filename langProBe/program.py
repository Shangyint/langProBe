import re
import dspy
from copy import deepcopy
from dsp.utils import deduplicate


#################################### Common Adapters ####################################


def CoTAdapter(signature):
    return dspy.ChainOfThought(signature)


def default_input_to_query(**kwargs):
    if len(kwargs) == 1:
        return list(kwargs.values())[0]
    else:
        raise ValueError(
            "Cannot convert multiple inputs to a query, please specify input_to_query."
        )


def RAGAdapter(
    signature, retriever=dspy.Retrieve(k=3), input_to_query=default_input_to_query
):
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


def SimplifiedBaleenAdapter(
    signature, query_gen_input=None, retriever=dspy.Retrieve(k=2), max_hops=2
):
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
            verified_signature = verified_signature.append(
                "context", dspy.InputField(desc="may contain relevant facts")
            )

            # remove the output field from the generate query signature
            # generate_query should use a default instruction rather than instruction from the original signature
            # FIXME (shangyin) fix the default signature.instructions
            input_fields = verified_signature.input_fields
            generate_query_signature = dspy.Signature(input_fields)
            generate_query_signature = generate_query_signature.append(
                "search_query", dspy.OutputField()
            )

            self.generate_query = [
                dspy.ChainOfThought(generate_query_signature)
                for _ in range(self.max_hops)
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


#################################### Archon Adapters ####################################


def ArchonGeneratorAdapter(signature, n=5):
    # https://github.com/ScalingIntelligence/Archon/blob/main/src/archon/completions/components/Generator.py
    # For dspy, n responses are generated with a single model now.
    # If desired, we can create a new module in dspy that uses multiple models to generate n responses.
    class ArchonGenerator(dspy.Module):
        def __init__(self):
            verified_signature = dspy.ensure_signature(signature)
            assert (
                len(verified_signature.output_fields) == 1
            ), "ArchonGenerator only supports a single output field"

            self.prog = dspy.ChainOfThought(verified_signature, n=n)
            self.output_field = list(verified_signature.output_fields.keys())[0]

        def forward(self, **kwargs):
            return self.prog(**kwargs)

        def get_responses(self, **kwargs):
            responses = self.prog(**kwargs).completions.__getattr__(self.output_field)
            return responses

    return ArchonGenerator()


def critic_responses_formatter(responses):
    if not isinstance(responses, list):
        dspy.logger.warning(
            "Responses of CriticGenerator should be a list of responses. "
        )
        responses = [responses]
    for i, response in enumerate(responses):
        responses[i] = f"[{i+1}] {response}"
    return "\n".join(responses)


class CriticGenerator(dspy.Signature):
    """
    Evaluate all responses based on their relevance to the instructions.
    All the responses should be included and evaluated using identifiers.
    You must include both strengths and weaknesses, even if there are more of one than the other.
    Do not include any preface or text after the critiques. Do not include any references to previous critiques within a critique.
    Start with the analysis for the first response and end with the analysis for the last response.
    """

    task_instructions = dspy.InputField(
        desc="The instructions to how the responses are generated."
    )
    responses = dspy.InputField(
        desc="The generated responses to critize. Each response will start with a numerical identifier in [], like [1].",
        format=critic_responses_formatter,
    )
    critics = dspy.OutputField(
        desc="The critic for each response. Structure each response's analysis as\
                                follows: [1]\nStrengths:\n- <strength #1>\n- <strength #2>\n- <strength #n>\
                                \nWeaknesses:\n- <weakness #1>\n- <weakness #2>\n- <weakness #n>\n\n"
    )


def ArchonCriticAdapter(signature, n=5):
    # https://github.com/ScalingIntelligence/Archon/blob/main/src/archon/completions/components/Critic.py
    # signature should be the signature to the original generator module

    class ArchonCritic(dspy.Module):
        def __init__(self):
            verified_signature = dspy.ensure_signature(signature)
            assert (
                len(verified_signature.output_fields) == 1
            ), "ArchonCritic only supports a single output field"
            self.signature = verified_signature

            self.instructions = verified_signature.instructions

            self.critic = dspy.Predict(CriticGenerator)

        def forward(self, responses):
            return self.critic(task_instructions=self.instructions, responses=responses)

        def get_critics(self, responses):
            critics = self.critic(
                task_instructions=self.instructions, responses=responses
            ).critics
            return critics

        def parse_critics(self, critics):
            segments = re.split(r"\[\d+\]", critics)
            pass  # we may not need to parse the critics for now

    return ArchonCritic()


class RankerGenerator(dspy.Signature):
    """
    I will provide you with responses, each indicated by a numerical identifier [].
    Rank the responses based on their relevance to the instruction"""

    task_instructions = dspy.InputField(
        desc="The instructions to how the responses are generated."
    )

    responses = dspy.InputField(
        desc="The responses to rank. Each response will start with a numerical identifier in [], like [1].",
        format=critic_responses_formatter,
    )

    ranking = dspy.OutputField(
        desc="The ranking of the responses. List the responses in descending order of relevance to the instructions.\
            Your output format should be a single line with [] > [], e.g., [4] > [2] > ... > [1]."
    )


def ArchonRankerAdapter(signature, n=5):
    # https://github.com/ScalingIntelligence/Archon/blob/main/src/archon/completions/components/prompts.py#L68

    class ArchonRanker(dspy.Module):
        def __init__(self):
            verified_signature = dspy.ensure_signature(signature)
            assert (
                len(verified_signature.output_fields) == 1
            ), "ArchonRanker only supports a single output field"
            self.signature = verified_signature
            self.instructions = verified_signature.instructions

            self.ranker = dspy.ChainOfThought(RankerGenerator)

        def forward(self, responses, **kwargs):
            if kwargs.get("critics", None):
                new_signature = self.ranker.extended_signature.append(
                    "critics",
                    dspy.InputField(
                        desc="The critics (strength/weakness) for each response."
                    ),
                )
                new_signature.instructions += (
                    "and their provided critiques of strengths and weaknesses."
                )

                return self.ranker(
                    new_signature=new_signature,
                    task_instructions=self.instructions,
                    responses=responses,
                    **kwargs,
                )
            else:
                return self.ranker(
                    task_instructions=self.instructions, responses=responses
                )

        def get_ranking(self, responses, **kwargs):
            raw_ranking = self.forward(responses, **kwargs).ranking
            ranks_str = re.findall(r"\[(\d+)\]", raw_ranking)
            ranks = [int(r) for r in ranks_str]
            if len(ranks) != len(responses):
                dspy.logger.warning(
                    "ArchonRanker: the number of responses and the number of ranks do not match."
                )
            return ranks

    return ArchonRanker()


# TODO(shangyin) new adapters from Archon to be added: Fuser, Verifier


def ArchonPipelineExample1(signature, n=5):
    class ArchonExample(dspy.Module):
        def __init__(self):
            verified_signature = dspy.ensure_signature(signature)
            assert (
                len(verified_signature.output_fields) == 1
            ), "ArchonExample only supports a single output field"
            self.signature = verified_signature

            self.generator = ArchonGeneratorAdapter(self.signature, n)
            self.critic = ArchonCriticAdapter(self.signature, n)
            self.ranker = ArchonRankerAdapter(self.signature, n)

        def forward(self, **kwargs):
            responses = self.generator.get_responses(**kwargs)
            print(responses)
            critics = self.critic.get_critics(responses)
            ranking = self.ranker.get_ranking(responses, critics=critics)
            return responses[ranking[0]]

    return ArchonExample()


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
    dspy.settings.lm.inspect_history(n=3)

    # ArchonPipelineExample1
    print("======== ArchonPipelineExample1 =========")
    archon_example = ArchonPipelineExample1("question -> answer")
    archon_example(question=question)
    dspy.settings.lm.inspect_history(n=3)
