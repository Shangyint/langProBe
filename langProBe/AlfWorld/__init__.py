from langProBe.AlfWorld.AlfWorld_data import AlfWorldBench
from langProBe.AlfWorld.AlfWorld_program import AlfWorldReAct, AlfWorldPredict, AlfWorldCoT
from langProBe.AlfWorld.AlfWorld_utils.alfworld_metric import alfworld_metric
from langProBe.benchmark import BenchmarkMeta
from langProBe.optimizers import OptimizerConfig
import dspy.teleprompt


ALFWORLD_OPTIMIZERS = [
    OptimizerConfig(
        optimizer=dspy.teleprompt.BootstrapFewShot,
        init_args=dict(max_errors=1000, max_labeled_demos=0, max_bootstrapped_demos=2),
        compile_args=dict(),
        langProBe_configs=dict(use_valset=False, name="BootstrapFewShot"),
    ),
    OptimizerConfig(
        optimizer=dspy.teleprompt.BootstrapFewShotWithRandomSearch,
        init_args=dict(max_errors=1000, max_labeled_demos=0, max_bootstrapped_demos=2, num_threads=8, num_candidate_programs=8),
        compile_args=dict(),
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
        ),
        langProBe_configs=dict(use_valset=True, name="MIPROv2"),
    ),
]


benchmark = [BenchmarkMeta(AlfWorldBench, [
    # AlfWorldReAct(),
    # AlfWorldPredict(),
    AlfWorldCoT()
], alfworld_metric, optimizers=ALFWORLD_OPTIMIZERS), ]
