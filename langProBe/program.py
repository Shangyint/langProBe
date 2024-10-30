import re
import dspy
from copy import deepcopy
from dsp.utils import deduplicate


#################################### Common Programs ####################################


def CoT(signature):
    return dspy.ChainOfThought(signature)


def default_input_to_query(**kwargs):
    if len(kwargs) == 1:
        return list(kwargs.values())[0]
    else:
        raise ValueError(
            "Cannot convert multiple inputs to a query, please specify input_to_query."
        )


class RAG(dspy.Module):
    def __init__(
        self,
        signature,
        retriever=dspy.Retrieve(k=3),
        input_to_query=default_input_to_query,
    ):
        self.retriver = retriever
        verified_signature = dspy.ensure_signature(signature)
        verified_signature = verified_signature.prepend(
            "context", dspy.InputField(desc="may contain relevant facts")
        )
        self.prog = dspy.ChainOfThought(verified_signature)
        self.input_to_query = input_to_query

    def forward(self, **kwargs):
        context = self.retriver(self.input_to_query(**kwargs)).passages
        pred = self.prog(context=context, **kwargs)
        return pred


class SimplifiedBaleen(dspy.Module):
    def __init__(
        self, signature, query_gen_input=None, retriever=dspy.Retrieve(k=2), max_hops=2
    ):
        """
        args:
            signature: The signature to the final generate module
            query_gen_input: a list of keywords to be used as input to the query generation module
            retriever: a retriever module to be used to retrieve relevant facts
            max_hops: the number of hops to be used in the simplified
            FIXME (shangyin) correctly handle query_gen_input
        """

        self.max_hops = max_hops
        self.retriever = retriever
        verified_signature = dspy.ensure_signature(signature)
        verified_signature = verified_signature.prepend(
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


#################################### Archon Programs ####################################

# Note Ranker and Fuser are equipped with self.get_prediction() method to return a Prediction object
# in the original signature

class ArchonGenerator(dspy.Module):
    # https://github.com/ScalingIntelligence/Archon/blob/main/src/archon/completions/components/Generator.py

    def __init__(self, signature, n=5):
        # For dspy, n responses are generated with a single model now.
        # If desired, we can create a new module in dspy that uses multiple models to generate n responses.
        verified_signature = dspy.ensure_signature(signature)
        assert (
            len(verified_signature.output_fields) == 1
        ), "ArchonGenerator only supports a single output field"

        self.prog = dspy.ChainOfThought(verified_signature, n=n)
        self.output_field = list(verified_signature.output_fields.keys())[0]

    def forward(self, **kwargs) -> dspy.Prediction:
        return self.prog(**kwargs)

    def get_responses(self, **kwargs) -> list[str]:
        responses = self.prog(**kwargs).completions.__getattr__(self.output_field)
        return responses

    def get_formatted_responses(self, **kwargs) -> str:
        responses = self.get_responses(**kwargs)
        return responses_formatter(responses)


def responses_formatter(responses):
    if not isinstance(responses, list):
        dspy.logger.warning(
            "Responses of CriticGenerator should be a list of responses. "
        )
        responses = [responses]
    for i, response in enumerate(responses):
        responses[i] = f"[{i+1}] {response}"
    return "\n".join(responses)


class FeedbackGeneratorSignature(dspy.Signature):
    """
    Evaluate all responses based on their relevance to the instructions.
    All the responses should be included and evaluated using identifiers.
    You must include both strengths and weaknesses, even if there are more of one than the other.
    Start with the analysis for the first response and end with the analysis for the last response.
    """

    task_instructions = dspy.InputField(
        desc="The instructions to how the responses are generated."
    )
    responses = dspy.InputField(
        desc="The generated responses to critize. Each response will start with a numerical identifier in [], like [1].",
    )
    feedback: list[str] = dspy.OutputField(
        desc="The feedback for each response. Discuss the strengths and weaknesses of each response."
    )


class ArchonCritic(dspy.Module):
    # https://github.com/ScalingIntelligence/Archon/blob/main/src/archon/completions/components/Critic.py

    def __init__(self, signature, n=5):
        # signature should be the signature to the original generator module
        verified_signature = dspy.ensure_signature(signature)
        assert (
            len(verified_signature.output_fields) == 1
        ), "ArchonCritic only supports a single output field"
        self.signature = verified_signature

        self.instructions = verified_signature.instructions

        self.feedback_gen = dspy.Predict(FeedbackGeneratorSignature)

    def forward(self, formatted_responses) -> dspy.Prediction:
        return self.feedback_gen(
            task_instructions=self.instructions, responses=formatted_responses
        )

    def get_feedback(self, formatted_responses: str) -> list[str]:
        return self.forward(formatted_responses).feedback


class RankerGeneratorSignature(dspy.Signature):
    """
    Rank the responses based on their relevance to the instruction"""

    task_instructions = dspy.InputField(
        desc="The instructions to how the responses are generated."
    )

    responses = dspy.InputField(
        desc="The responses to rank. Each response will start with a numerical identifier in [], like [1].",
    )

    ranking: list[int] = dspy.OutputField(
        desc="The ranking of the responses. List the responses in descending order of relevance to the instructions."
    )


class ArchonRanker(dspy.Module):
    # https://github.com/ScalingIntelligence/Archon/blob/main/src/archon/completions/components/prompts.py#L68
    def __init__(self, signature, n=5, use_critic=False):
        verified_signature = dspy.ensure_signature(signature)
        assert (
            len(verified_signature.output_fields) == 1
        ), "ArchonRanker only supports a single output field"
        self.signature = verified_signature
        self.instructions = verified_signature.instructions

        ranker_signature = RankerGeneratorSignature
        if use_critic:
            ranker_signature = ranker_signature.append(
                "feedback",
                dspy.InputField(
                    desc="The feedback (strength/weakness) for each response."
                ),
            )
            ranker_signature.instructions += (
                "and their provided critiques of strengths and weaknesses."
            )

        self.ranker = dspy.ChainOfThought(ranker_signature)

    def forward(self, formatted_responses: str, **kwargs):
        return self.ranker(
            task_instructions=self.instructions, responses=formatted_responses, **kwargs
        )

    def get_ranking(self, formatted_responses: str, **kwargs) -> list[int]:
        return self.forward(formatted_responses, **kwargs).ranking
    
    def get_prediction(self, responses: list[str], **kwargs) -> dspy.Prediction:
        formatted_responses = responses_formatter(responses)
        ranking = self.get_ranking(formatted_responses, **kwargs)
        top_response = responses[ranking[0]]
        pred = dspy.Prediction()
        pred.__setattr__(list(self.signature.output_fields.keys())[0], top_response)
        return pred


class FuserGeneratorSignature(dspy.Signature):
    """
    Compile a list of responses into a single high-quality response.
    For each response, you should include its strengths and avoid its weaknesses."""

    task_instructions = dspy.InputField(
        desc="The instructions to how the responses are generated."
    )

    responses = dspy.InputField(
        desc="The responses to synthesize.",
    )

    final_response = dspy.OutputField(
        desc="The final response, synthesized from the input responses."
    )


class ArchonFuser(dspy.Module):
    def __init__(self, signature, use_critic=False):
        verified_signature = dspy.ensure_signature(signature)
        assert (
            len(verified_signature.output_fields) == 1
        ), "ArchonFuser only supports a single output field"
        self.signature = verified_signature
        self.instructions = verified_signature.instructions

        fuser_signature = FuserGeneratorSignature
        if use_critic:
            fuser_signature = fuser_signature.append(
                "feedback",
                dspy.InputField(
                    desc="The feedback (strength/weakness) for each response."
                ),
            )
            fuser_signature.instructions += "For each response, we also provide critiques of strengths and weaknesses."

        self.fuser = dspy.ChainOfThought(fuser_signature)

    def forward(self, formatted_responses: str, **kwargs):
        return self.fuser(
            task_instructions=self.instructions, responses=formatted_responses, **kwargs
        )

    def get_response(self, formatted_responses: str, **kwargs) -> str:
        return self.forward(formatted_responses, **kwargs).final_response
    
    def get_prediction(self, formatted_responses: str, **kwargs) -> dspy.Prediction:
        final_response = self.get_response(formatted_responses, **kwargs)
        pred = dspy.Prediction()
        pred.__setattr__(list(self.signature.output_fields.keys())[0], final_response)
        return pred


# TODO(shangyin) new adapters from Archon to be added: Verifier

#################################### Archon Example Programs ####################################


class GeneratorCriticRanker(dspy.Module):
    def __init__(self, signature, n=5):
        verified_signature = dspy.ensure_signature(signature)
        assert (
            len(verified_signature.output_fields) == 1
        ), "ArchonExample only supports a single output field"
        self.signature = verified_signature

        self.generator = ArchonGenerator(self.signature, n)
        self.critic = ArchonCritic(self.signature, n)
        self.ranker = ArchonRanker(self.signature, n, use_critic=True)

    def forward(self, **kwargs):
        responses = self.generator.get_responses(**kwargs)
        formatted_responses = responses_formatter(responses)
        feedback = self.critic.get_feedback(formatted_responses)
        return self.ranker.get_prediction(responses, feedback=feedback)


class GeneratorCriticFuser(dspy.Module):
    def __init__(self, signature, n=5):
        verified_signature = dspy.ensure_signature(signature)
        assert (
            len(verified_signature.output_fields) == 1
        ), "GeneratorCriticFuser only supports a single output field"
        self.signature = verified_signature

        self.generator = ArchonGenerator(self.signature, n)
        self.critic = ArchonCritic(self.signature, n)
        self.fuser = ArchonFuser(self.signature, use_critic=True)

    def forward(self, **kwargs):
        formatted_responses = self.generator.get_formatted_responses(**kwargs)
        feedback = self.critic.get_feedback(formatted_responses)
        return self.fuser.get_prediction(formatted_responses, feedback=feedback)


class GeneratorRanker(dspy.Module):
    def __init__(self, signature, n=5):
        verified_signature = dspy.ensure_signature(signature)
        assert (
            len(verified_signature.output_fields) == 1
        ), "GeneratorRanker only supports a single output field"
        self.signature = verified_signature

        self.generator = ArchonGenerator(self.signature, n)
        self.ranker = ArchonRanker(self.signature, use_critic=False)

    def forward(self, **kwargs):
        responses = self.generator.get_responses(**kwargs)
        return self.ranker.get_top_ranked_prediction(responses)


class GeneratorFuser(dspy.Module):
    def __init__(self, signature, n=5):
        verified_signature = dspy.ensure_signature(signature)
        assert (
            len(verified_signature.output_fields) == 1
        ), "GeneratorFuser only supports a single output field"
        self.signature = verified_signature

        self.generator = ArchonGenerator(self.signature, n)
        self.fuser = ArchonFuser(self.signature, use_critic=False)

    def forward(self, **kwargs):
        formatted_responses = self.generator.get_formatted_responses(**kwargs)
        return self.fuser(formatted_responses)


if __name__ == "__main__":
    # Example usage
    dspy.configure(
        lm=dspy.LM("openai/gpt-4o-mini"),
        # example rm for RAG w. passages from wikipedia dump
        rm=dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts"),
    )

    question = "What is the capital of France?"
    context = "France is a country in Europe."

    # CoT
    print("======== CoT =========")
    cot = CoT("question, context -> answer")
    cot(question=question, context=context)
    dspy.settings.lm.inspect_history()

    # RAG
    print("======== RAG =========")
    rag = RAG("question -> answer")
    rag(question=question)
    dspy.settings.lm.inspect_history()

    # SimplifiedBaleen
    print("======== SimplifiedBaleen =========")
    simplified_baleen = SimplifiedBaleen("question -> answer")
    simplified_baleen(question=question)
    dspy.settings.lm.inspect_history(n=3)

    # GeneratorCriticRanker
    print("======== GeneratorCriticRanker =========")
    archon_example = GeneratorCriticRanker("question -> answer")
    archon_example(question=question)
    dspy.settings.lm.inspect_history(n=3)

    # GeneratorRanker
    print("======== GeneratorRanker =========")
    generator_ranker = GeneratorRanker("question -> answer")
    generator_ranker(question=question)
    dspy.settings.lm.inspect_history(n=3)

    # GeneratorCriticFuser
    print("======== GeneratorCriticFuser =========")
    generator_critic_fuser = GeneratorCriticFuser("question -> answer")
    generator_critic_fuser(question=question)
    dspy.settings.lm.inspect_history(n=3)

    # GeneratorFuser
    print("======== GeneratorFuser =========")
    generator_fuser = GeneratorFuser("question -> answer")
    generator_fuser(question=question)
    dspy.settings.lm.inspect_history(n=3)
