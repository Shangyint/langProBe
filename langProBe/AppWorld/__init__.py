import os

from .AppWorld_data import AppWorldBench
from .AppWorld_utils.appworld_metric import appworld_metric
from .AppWorld_program import AppWorldReact

from langProBe.benchmark import BenchmarkMeta

from langProBe.optimizers import DEFAULT_OPTIMIZERS, OptimizerConfig
import dspy
import dspy.teleprompt

appworld_teacher = AppWorldReact(add_few_shot=True)
appworld_teacher._name = "AppWorldReactAugmented"
appworld_student = AppWorldReact(add_few_shot=False)
appworld_student._name = "AppWorldReact"

APPWORLD_OPTIMIZERS = [
    OptimizerConfig(
        optimizer=dspy.teleprompt.BootstrapFewShot,
        init_args=dict(max_errors=1000, max_labeled_demos=0, max_bootstrapped_demos=2),
        compile_args=dict(teacher=appworld_teacher),
        langProBe_configs=dict(use_valset=False, name="BootstrapFewShot"),
    ),
    OptimizerConfig(
        optimizer=dspy.teleprompt.BootstrapFewShotWithRandomSearch,
        init_args=dict(
            max_errors=1000,
            max_labeled_demos=0,
            max_bootstrapped_demos=2,
            num_threads=8,
            num_candidate_programs=8,
        ),
        compile_args=dict(teacher=appworld_teacher),
        langProBe_configs=dict(
            use_valset=True, name="BootstrapFewShotWithRandomSearch"
        ),
    ),
    OptimizerConfig(
        optimizer=dspy.teleprompt.MIPROv2,
        init_args=dict(max_errors=1000, auto="medium", num_threads=8),
        compile_args=dict(
            requires_permission_to_run=False,
            num_trials=20,
            max_bootstrapped_demos=2,
            max_labeled_demos=0,
            teacher=appworld_teacher,
        ),
        langProBe_configs=dict(use_valset=True, name="MIPROv2"),
    ),
]


benchmark = [
    BenchmarkMeta(
        AppWorldBench,
        [appworld_student, appworld_teacher],
        appworld_metric,
        dataset_mode="full",
        optimizers=APPWORLD_OPTIMIZERS,
    )
]

os.environ["APPWORLD_ROOT"] = os.path.join(os.getcwd(), "langProBe", "AppWorld")
