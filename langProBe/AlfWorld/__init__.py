from langProBe.AlfWorld.AlfWorld_data import AlfWorldBench
from langProBe.AlfWorld.AlfWorld_program import AlfWorldReactSolver, AlfWorldSolver, AlfWorldSolverWithThought
from langProBe.AlfWorld.AlfWorld_utils.alfworld_metric import alfworld_metric
from langProBe.benchmark import BenchmarkMeta

benchmark = [BenchmarkMeta(AlfWorldBench, [AlfWorldReactSolver, AlfWorldSolver, AlfWorldSolverWithThought], alfworld_metric)]
