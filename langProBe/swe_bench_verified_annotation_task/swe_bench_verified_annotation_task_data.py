import dspy.teleprompt
from ..benchmark import Benchmark

import pandas as pd

import dspy

import random
random.seed(1, version=2)

class SWEBenchVerifiedAnnotationTaskBench(Benchmark):
    def process_df_to_examples(self, df):
        common_keys = ['instance_id', 'task_batch_name', 'base_commit', 'patch', 'test_patch', 'problem_statement', 'hints_text', 'created_at', 'version', 'FAIL_TO_PASS', 'PASS_TO_PASS', 'environment_setup_commit', 'repo']
        
        individual_annotation_keys = ['user_id', 'difficulty', 'difficulty_notes', 'underspecified', 'underspecified_notes', 'false_negative', 'false_negative_notes', 'other_major_issues', 'other_notes', 'annotator_confidence', 'claimed_at', 'updated_at', 'submitted_at', 'reviews', 'comments', 'problematic', 'underspecified_problematic', 'false_negative_problematic', ]
        score_keys = ['underspecified', 'false_negative']
        for score_key in score_keys:
            score_key in individual_annotation_keys
        dataset = []

        instance_ids = df['instance_id'].unique()
        for instance_id in instance_ids:
            dfx = df[df['instance_id'] == instance_id]
            assert len(dfx) == 3
            row_1 = dfx.iloc[0]
            instance_example = {}
            for k in common_keys:
                instance_example[k] = str(row_1[k])
            
            for k in individual_annotation_keys:
                if k in score_keys:
                    l = [int(x) for x in dfx[k].tolist()]
                    min_l = min(l)
                    max_l = max(l)
                    instance_example[k] = [str(x) for x in range(min_l, max_l + 1)]
                else:
                    instance_example[k] = [str(x) for x in dfx[k].tolist()]
            
            dataset.append(dspy.Example(**instance_example).with_inputs('patch', 'test_patch', 'problem_statement', 'FAIL_TO_PASS', 'repo'))
        
        return dataset

    def init_dataset(self):
        df_test = pd.read_csv("langProBe/swe_bench_verified_annotation_task/SweBenchVerifiedAnnotationTaskDataset/test_split.csv")
        self.test_set = self.process_df_to_examples(df_test)

        assert len(self.test_set) == 400
        
        df_train_val = pd.read_csv("langProBe/swe_bench_verified_annotation_task/SweBenchVerifiedAnnotationTaskDataset/trainval_split.csv")
        self.dataset = self.process_df_to_examples(df_train_val)

        assert len(self.dataset) == 1289, len(self.dataset)
