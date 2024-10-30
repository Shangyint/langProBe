import dspy.teleprompt

import re
import dspy

class UnderspecifiedSignature(dspy.Signature):
    """You are considering an issue from an open source Python repository. The issue has an associated test_patch that provides a set of tests to check whether the issue is resolved. The issue also has a corresponding gold_patch that provides a fix for the issue. Your task is to evaluate if this issue should be included in a question bank to benchmark coding ability. To evaluate an engineer's coding ability, the issue will be provided to them without the gold_patch and they will be tasked with writing code to resolve the issue. The test_patch will then be used to check whether the engineer's solution passes the tests in the test_patch. However, to be a fair evaluation, the issue must have a well-specified issue description that is clear and unambiguous about what is required for a successful solution. The engineer should have complete information about what is required to resolve the issue, just by reading the issue description. Please read the issue description and the test_patch carefully.

Imagine that you are an experienced software engineer who has been instructed to solve the issue. You have full access to the codebase, and can see the issue description as it is below. But you are not able to ask for clarification and would need to work exclusively from this information. Is the issue description well-specified enough for a meaningful attempt at a solution? Please check carefully and remember that the engineer attempting this issue will not have access to the original gold_patch or the test_patch.

Provide a score from 0 to 3 based on the following criteria:
0: The issue is well-specified and it is clear what is required for a successful solution.
1: There are some blanks to fill in about the issue, but there is a sensible interpretation of what is required for a successful solution.
2: The issue is vague and there is room for ambiguity. It is unclear what a successful solution would look like.
3: It is almost impossible to understand what you are being asked to do without further information.

Reason carefully about specific functions and classes before giving your answer."""
    # TODO: Map 'problem_statement' to 'issue_description'
    # TODO: Map 'FAIL_TO_PASS' to 'test_names'
    # TODO: Map 'patch' to 'gold_patch'
    repo: str = dspy.InputField(desc="The repository name containing the issue")
    issue_description: str = dspy.InputField(desc="The issue description")
    gold_patch: str = dspy.InputField(desc="The gold patch")
    test_patch: str = dspy.InputField(desc="The test patch")
    test_names: str = dspy.InputField(desc="The names of the tests in the test patch that will be used to evaluate the solution")
    # reasoning = dspy.OutputField(str, "Let's think step by step about the issue description and the test patch to determine if the tests are well-scoped")
    score: str = dspy.OutputField(desc="The score from 0 to 3 based on the criteria")

class UnderspecifiedAnnotationGenerator(dspy.Module):
    def __init__(self):
        self.underspcification_predictor = dspy.ChainOfThought(UnderspecifiedSignature)

    def forward(self, patch, test_patch, problem_statement, FAIL_TO_PASS, repo):
        underspecification_output = self.underspcification_predictor(
            repo=repo,
            issue_description=problem_statement,
            gold_patch=patch,
            test_patch=test_patch,
            test_names=FAIL_TO_PASS,
        )
        
        underspecification_output['score'] = str(int(re.split(r'[^0-9]+', underspecification_output['score'])[0]))

        output = {}
        
        for k, v in underspecification_output.items():
            output["underspecification_" + k] = v
        
        return dspy.Prediction(**output)
