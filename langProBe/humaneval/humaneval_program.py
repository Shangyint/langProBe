import dspy

from itertools import chain
from .humaneval_utils import post_process_tests, post_process_code

NUM_SAMPLES = 20
TEMPARATURE_BASE = 0.7
TEMPARATURE_STEP = 0.01
NUM_TESTS = 5

# Code Generator


class CodeProblem(dspy.Signature):
    prompt = dspy.InputField(format=str)
    code = dspy.OutputField()


class CoT(dspy.Module):
    def __init__(self):
        self.prog = dspy.ChainOfThought("prompt -> code")

    def forward(self, prompt, **kargs):
        pred = self.prog(prompt=prompt)
        return pred


# Test Generator


class GenerateTests(dspy.Signature):
    prompt = dspy.InputField(format=str)
    tests = dspy.OutputField(
        desc="Executable tests using assert, you can use the one in prompts. \
                             Do not put tests in another local function, directly write them."
    )


class ExtractTests(dspy.Signature):
    """
    Extract tests/examples from the prompt, and convert them to executable asserts. DO NOT INVENT YOUR OWN TESTS!
    """

    prompt = dspy.InputField(format=str)
    tests = dspy.OutputField(
        desc="Executable tests using assert, you should directly extract the test from the prompt. \
                             Do not invent your own tests!"
    )


def generate_tests(prompt):
    test_gen = dspy.ChainOfThought(GenerateTests)
    tests = test_gen(prompt=prompt)
    return tests.tests


def extract_tests(prompt):
    test_gen = dspy.ChainOfThought(ExtractTests)
    tests = test_gen(prompt=prompt)
    return tests.tests


def generate_tests_lists(prompt):
    test_gen = dspy.ChainOfThought(GenerateTests)
    tests = []
    for i in range(NUM_TESTS):
        raw_tests = test_gen(
            prompt=prompt, config=dict(temperature=00.7 + (0.1 * i))
        ).tests
        tests.append(post_process_tests(post_process_code(raw_tests)))
    result = list(chain(*tests))
    return result


class MultiChain(dspy.Module):
    def __init__(self, num_chain=5):
        self.num_chain = num_chain
        self.reasoning_generator = dspy.ChainOfThought(
            "prompt -> code", n=self.num_chain
        )
        self.prog = dspy.MultiChainComparison("prompt -> code", M=self.num_chain)

    def forward(self, prompt, **kargs):
        completions = self.reasoning_generator(prompt=prompt)
        pred = self.prog(completions.completions, prompt=prompt)
        return pred
