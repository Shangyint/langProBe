import dspy.teleprompt
from langProBe import dspy_program
import re
import dspy


class EvaluationValiditySignature(dspy.Signature):
    """You are now considering an issue from an open source Python repository. The issue has an associated test_patch that provides a set of tests to check whether the issue is resolved. The issue also has a corresponding gold_patch that provides a solution to the issue. Your task is to evaluate if this issue should be included in a benchmark for coding ability. The issue will be provided to an engineer (without the gold_patch) and the engineer will be asked to write code to resolve the issue. The test_patch will then be used to check whether the engineer's solution passes the tests in the test_patch. However, to be included in the benchmark, the issue must have tests that are correctly scoped to the issue, such that all reasonable solutions to the issue should pass the tests. The tests should not rely on any details that are not present in the issue description. The tests must check for a solution exactly as described in the issue description, and do not test any other unrelated functionality, like specifics of the gold_patch.

    Given a new candidate solution, we intend to use the test_patch to check whether the solution correctly resolves the issue. However, remember that these tests were written with a particular solution in mind (the gold_patch) so they may be ill-suited for evaluating other valid solutions. We would like to know if the tests are correctly scoped to identify all reasonable solutions to the issue, or if the tests rely on narrow details that would unfairly penalize a new solution that is otherwise correct. In other words, our setup only works if the tests do not rely on any details that are not present in the issue description. We find that the most common problems occur when there are subtle discrepancies between the tests and the issue text, such as the tests relying on a new function, variable name, or error message that were introduced in the gold_patch but is not mentioned or differs from the issue description. Please check carefully for such discrepancies, and remember that the engineer attempting this issue will not have access to the original gold_patch or the test_patch.

    Are the tests well-scoped such that all reasonable solutions to the issue should pass the tests? Provide a score from 0 to 3 based on the following criteria:
    0: The tests perfectly cover all possible solutions.
    1: The tests cover the majority of correct solutions, however some unusual solutions may be missed.
    2: The tests work but some perfectly reasonable solutions may be missed by the tests.
    3: The tests are too narrow/broad or they look for something different than what the issue is about.

    Reason carefully about specific functions and classes before giving your answer."""

    # TODO: 1 We later clarified to annotators that they should assume embedded images in the Issue Description would be visible.
    # TODO: Map 'problem_statement' to 'issue_description'
    # TODO: Map 'FAIL_TO_PASS' to 'test_names'
    # TODO: Map 'patch' to 'gold_patch'
    repo: str = dspy.InputField(desc="The repository name containing the issue")
    issue_description: str = dspy.InputField(desc="The issue description")
    gold_patch: str = dspy.InputField(desc="The gold patch")
    test_patch: str = dspy.InputField(desc="The test patch")
    test_names: str = dspy.InputField(
        desc="The names of the tests in the test patch that will be used to evaluate the solution"
    )
    # reasoning = dspy.OutputField(str, "Let's think step by step about the issue description and the test patch to determine if the tests are well-scoped")
    score: str = dspy.OutputField(
        desc="The score from 0 to 3 based on the criteria", format=lambda x: str(x)
    )


class EvaluationValidityPredict(dspy.Module):
    def __init__(self):
        self.evaluation_validity_predictor = dspy_program.Predict(
            EvaluationValiditySignature
        )

    def forward(self, patch, test_patch, problem_statement, FAIL_TO_PASS, repo):
        evaluation_validity_output = self.evaluation_validity_predictor(
            repo=repo,
            issue_description=problem_statement,
            gold_patch=patch,
            test_patch=test_patch,
            test_names=FAIL_TO_PASS,
        )

        evaluation_validity_output["score"] = str(
            int(re.split(r"[^0-9]+", evaluation_validity_output["score"])[0])
        )

        output = {}

        for k, v in evaluation_validity_output.items():
            output["evaluation_validity_" + k] = v

        return dspy.Prediction(**output)


class EvaluationValidityCoT(dspy.Module):
    def __init__(self):
        self.evaluation_validity_predictor = dspy.ChainOfThought(
            EvaluationValiditySignature
        )

    def forward(self, patch, test_patch, problem_statement, FAIL_TO_PASS, repo):
        evaluation_validity_output = self.evaluation_validity_predictor(
            repo=repo,
            issue_description=problem_statement,
            gold_patch=patch,
            test_patch=test_patch,
            test_names=FAIL_TO_PASS,
        )

        evaluation_validity_output["score"] = str(
            int(re.split(r"[^0-9]+", evaluation_validity_output["score"])[0])
        )

        output = {}

        for k, v in evaluation_validity_output.items():
            output["evaluation_validity_" + k] = v

        return dspy.Prediction(**output)


class EvaluationValidityGeneratorCriticRanker(dspy.Module):
    def __init__(self):
        self.evaluation_validity_predictor = dspy_program.GeneratorCriticRanker(
            EvaluationValiditySignature
        )

    def forward(self, patch, test_patch, problem_statement, FAIL_TO_PASS, repo):
        evaluation_validity_output = self.evaluation_validity_predictor(
            repo=repo,
            issue_description=problem_statement,
            gold_patch=patch,
            test_patch=test_patch,
            test_names=FAIL_TO_PASS,
        )

        evaluation_validity_output["score"] = str(
            int(re.split(r"[^0-9]+", evaluation_validity_output["score"])[0])
        )

        output = {}

        for k, v in evaluation_validity_output.items():
            output["evaluation_validity_" + k] = v

        return dspy.Prediction(**output)


class EvaluationValidityGeneratorCriticFuser(dspy.Module):
    def __init__(self):
        self.evaluation_validity_predictor = dspy_program.GeneratorCriticFuser(
            EvaluationValiditySignature
        )

    def forward(self, patch, test_patch, problem_statement, FAIL_TO_PASS, repo):
        evaluation_validity_output = self.evaluation_validity_predictor(
            repo=repo,
            issue_description=problem_statement,
            gold_patch=patch,
            test_patch=test_patch,
            test_names=FAIL_TO_PASS,
        )

        evaluation_validity_output["score"] = str(
            int(re.split(r"[^0-9]+", evaluation_validity_output["score"])[0])
        )

        output = {}

        for k, v in evaluation_validity_output.items():
            output["evaluation_validity_" + k] = v

        return dspy.Prediction(**output)
