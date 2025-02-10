from dataclasses import dataclass
from typing import Any, Callable, Type
import dspy
import dspy.teleprompt
from functools import partial


import copy
import random
import numpy as np
from dspy.teleprompt import BootstrapFewShot, LabeledFewShot, MIPROv2

from dspy.evaluate.evaluate import Evaluate
import dspy


class BootstrapFewShotInfer(BootstrapFewShot):
    def __init__(
        self,
        num_candidates=5,
        num_rules=5,
        num_threads=8,
        teacher_settings=None,
        **kwargs,
    ):
        super().__init__(teacher_settings=teacher_settings, **kwargs)
        self.num_candidates = num_candidates
        self.num_rules = num_rules
        self.num_threads = num_threads
        self.rules_induction_program = RulesInductionProgramINFER(
            num_rules, teacher_settings=teacher_settings
        )
        self.metric = kwargs.get("metric")
        self.max_errors = kwargs.get("max_errors", 5)

    def compile(self, student, *, teacher=None, trainset, valset=None):
        super().compile(student, teacher=teacher, trainset=trainset)
        if valset is None:
            train_size = int(0.8 * len(trainset))
            trainset, valset = trainset[:train_size], trainset[train_size:]
        original_program = copy.deepcopy(self.student)
        all_predictors = [
            p for p in original_program.predictors() if hasattr(p, "signature")
        ]
        instructions_list = [p.signature.instructions for p in all_predictors]

        best_score = -np.inf
        best_program = None

        for candidate_idx in range(self.num_candidates):
            candidate_program = copy.deepcopy(original_program)
            candidate_predictors = [
                p for p in candidate_program.predictors() if hasattr(p, "signature")
            ]
            for i, predictor in enumerate(candidate_predictors):
                predictor.signature.instructions = instructions_list[i]
            for i, predictor in enumerate(candidate_predictors):
                rules = self.induce_natural_language_rules(predictor, trainset)
                predictor.signature.instructions = instructions_list[i]
                self.update_program_instructions(predictor, rules)
            score = self.evaluate_program(candidate_program, valset)
            if score > best_score:
                best_score = score
                best_program = candidate_program
                print(
                    f"New best candidate (Candidate {candidate_idx+1}) with score {best_score}"
                )
        print("Final best score:", best_score)
        self.student = best_program
        return best_program

    def induce_natural_language_rules(self, predictor, trainset):
        demos = self.get_predictor_demos(trainset, predictor)
        signature = predictor.signature
        while True:
            examples_text = self.format_examples(demos, signature)
            try:
                natural_language_rules = self.rules_induction_program(examples_text)
                break
            except Exception as e:
                print("entereing here")
                print(len(demos))

                if (
                    isinstance(e, ValueError)
                    or e.__class__.__name__ == "BadRequestError"
                    or "ContextWindowExceededError" in str(e)
                ):
                    if len(demos) > 1:
                        demos = demos[:-1]
                    else:
                        natural_language_rules = ""
                        raise RuntimeError(
                            "Failed to generate natural language rules: A single example could not fit in context."
                        ) from e
        return natural_language_rules

    def update_program_instructions(self, predictor, natural_language_rules):
        predictor.signature.instructions = (
            f"{predictor.signature.instructions}\n\n"
            f"Please apply the following rules when making your prediction:\n{natural_language_rules}"
        )

    def format_examples(self, demos, signature):
        examples_text = ""
        for demo in demos:
            input_fields = {
                k: v for k, v in demo.items() if k in signature.input_fields
            }
            output_fields = {
                k: v for k, v in demo.items() if k in signature.output_fields
            }
            input_text = "\n".join(f"{k}: {v}" for k, v in input_fields.items())
            output_text = "\n".join(f"{k}: {v}" for k, v in output_fields.items())
            examples_text += f"Example:\n{input_text}\n{output_text}\n\n"
        return examples_text

    def get_predictor_demos(self, trainset, predictor):
        signature = predictor.signature
        return [
            {
                key: value
                for key, value in example.items()
                if key in signature.input_fields or key in signature.output_fields
            }
            for example in trainset
        ]

    def evaluate_program(self, program, dataset):
        evaluate = Evaluate(
            devset=dataset,
            metric=self.metric,
            num_threads=self.num_threads,
            max_errors=self.max_errors,
            display_table=False,
            display_progress=True,
            return_all_scores=True,
        )
        score, _ = evaluate(program, metric=self.metric)
        return score


class RulesInductionProgramINFER(dspy.Module):
    def __init__(self, num_rules, teacher_settings=None, verbose=False):
        super().__init__()
        docstring = f"""Given a set of examples, extract a set of {num_rules} concise and non-redundant natural language rules that explain the patterns in the data. These rules should be specific and actionable, providing clear guidance for performing the task."""

        class CustomRulesInduction(dspy.Signature):
            __doc__ = docstring
            examples_text = dspy.InputField(desc="Text containing examples")
            natural_language_rules = dspy.OutputField(
                desc="Induced natural language rules"
            )

        self.rules_induction = dspy.ChainOfThought(CustomRulesInduction)
        self.verbose = verbose
        self.teacher_settings = teacher_settings or {}

    def forward(self, examples_text):
        original_temp = dspy.settings.lm.kwargs.get("temperature", 0.7)
        if self.teacher_settings:
            with dspy.settings.context(**self.teacher_settings):
                print("Using teacher settings")
                print(dspy.settings.lm.model)
                dspy.settings.lm.kwargs["temperature"] = random.uniform(0.9, 1.0)
                print(dspy.settings.lm.kwargs["temperature"])
                prediction = self.rules_induction(examples_text=examples_text)
        else:
            # print('Using default DSPy settings')
            # print(dspy.settings.lm.model)
            dspy.settings.lm.kwargs["temperature"] = random.uniform(0.9, 1.0)
            prediction = self.rules_induction(examples_text=examples_text)
            dspy.settings.lm.kwargs["temperature"] = original_temp
        natural_language_rules = prediction.natural_language_rules.strip()
        if self.verbose:
            print(natural_language_rules)
        return natural_language_rules


@dataclass
class OptimizerConfig:
    optimizer: Type[dspy.teleprompt.Teleprompter]
    init_args: dict
    compile_args: dict
    langProBe_configs: dict
    name: str

    def __str__(self):
        return f"""
[[
    Optimizer: {self.name} ({self.optimizer})
    init_args: {self.init_args}
    compile_args: {self.compile_args}
    langProBe_configs: {self.langProBe_configs}
]]
        """

    def __repr__(self):
        return self.__str__()


# Optimizer configuration formats:
DEFAULT_OPTIMIZERS = [
    OptimizerConfig(
        optimizer=dspy.teleprompt.BootstrapFewShot,
        init_args=dict(max_errors=5000, max_labeled_demos=2),
        compile_args=dict(),
        langProBe_configs=dict(use_valset=False, save_candidate_score=False),
        name="BootstrapFewShot",
    ),
    OptimizerConfig(
        optimizer=dspy.teleprompt.BootstrapFewShotWithRandomSearch,
        init_args=dict(max_errors=5000, max_labeled_demos=2, num_threads=16),
        compile_args=dict(),
        langProBe_configs=dict(use_valset=True, save_candidate_score=True),
        name="BootstrapFewShotWithRandomSearch",
    ),
    OptimizerConfig(
        optimizer=dspy.teleprompt.MIPROv2,
        init_args=dict(max_errors=5000, auto="medium", num_threads=16),
        compile_args=dict(
            requires_permission_to_run=False,
            num_trials=20,
            max_bootstrapped_demos=4,
            max_labeled_demos=2,
        ),
        langProBe_configs=dict(
            use_valset=True,
            save_candidate_score=True,
        ),
        name="MIPROv2-lite",
    ),
    OptimizerConfig(
        optimizer=dspy.teleprompt.MIPROv2,
        init_args=dict(max_errors=5000, num_threads=16, num_candidates=12),
        compile_args=dict(
            requires_permission_to_run=False,
            num_trials=50,
            max_bootstrapped_demos=4,
            max_labeled_demos=2,
            minibatch_size=35,
            minibatch_full_eval_steps=5,
        ),
        langProBe_configs=dict(
            use_valset=True,
            save_candidate_score=True,
        ),
        name="MIPROv2",
    ),
    OptimizerConfig(
        optimizer=BootstrapFewShotInfer,
        init_args=dict(max_errors=5000, num_candidates=10, num_rules=10, num_threads=8),
        compile_args=dict(),
        langProBe_configs=dict(use_valset=True),
        name="RuleInfer-lite",
    ),
    OptimizerConfig(
        optimizer=BootstrapFewShotInfer,
        init_args=dict(max_errors=5000, num_candidates=10, num_rules=20, num_threads=8),
        compile_args=dict(),
        langProBe_configs=dict(use_valset=True),
        name="RuleInfer",
    ),
]


def update_optimizer_from_list(
    optimizer_list: list[OptimizerConfig], optimizer: OptimizerConfig
) -> list[OptimizerConfig]:
    new_optimizer_list = []
    for optimizer_config in optimizer_list:
        if optimizer.optimizer == optimizer_config.optimizer:
            new_optimizer_list.append(optimizer)
        else:
            new_optimizer_list.append(optimizer_config)
    return new_optimizer_list


def create_optimizer(
    optimizer_config: OptimizerConfig, metric, num_threads=None
) -> tuple[Callable, dict]:
    name = optimizer_config.name
    optimizer = optimizer_config.optimizer
    init_args = optimizer_config.init_args
    if num_threads and "num_threads" in init_args:
        init_args["num_threads"] = num_threads
    compile_args = optimizer_config.compile_args
    langProBe_configs = optimizer_config.langProBe_configs | {"name": name}
    optimizer = optimizer(metric=metric, **init_args)
    return partial(optimizer.compile, **compile_args), langProBe_configs
