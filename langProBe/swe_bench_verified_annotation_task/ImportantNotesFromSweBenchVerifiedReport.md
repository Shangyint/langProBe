Total 93 independent annotators
nnotated 1699 testcases from SWE-bench
Labels go from [0, 3], with 0 being no or minor issue, 3 being severe
Difficulty annotated as "How long will a developer take to solve"
Freeform text for "other major issues"

Team of OpenAI engineers handlabeled 50 samples to high degree of confidence.
Each annotator had to pass onboarding test.

In the final dataset, each sample labeled 3 times by separate annotators
Take the highest severeity label among 3 as the final label

Annotation Criteria:
1. Are the tasks well specified
2. How valid are the evaluation criteria: Could the FAIL_TO_PASS tests fail even with a valid solution?
3. (Not used for dataset filtering) How long will a developer take to solve the task?

Final dataset: filter out any sample from the original test set where either task 1 or task 2 have ensemble 
label of 2 or above in severity

Also filter out samples with other major issues flagged

Include as many samples with difficulty 1-4 and >4 hours as possible, and then randomly sample the remainder to select 500 samples.

Shown samples in the report include "Problem Statement", "Are the tasks well specified", "FAIL_TO_PASS test (Only showing lines added during the original PR for brevity)"