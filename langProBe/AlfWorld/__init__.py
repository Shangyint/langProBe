from langProBe.AlfWorld.AlfWorld_data import AlfWorldBench
from langProBe.AlfWorld.AlfWorld_program import AlfWorldReAct, AlfWorldPredict, AlfWorldCoT
from langProBe.AlfWorld.AlfWorld_utils.alfworld_metric import alfworld_metric
from langProBe.benchmark import BenchmarkMeta

benchmark = [BenchmarkMeta(AlfWorldBench, [AlfWorldReAct, AlfWorldPredict, AlfWorldCoT], alfworld_metric)]
