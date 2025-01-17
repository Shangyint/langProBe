from .irera_data import IReRaBench
from .irera_program import IReRaPredict, IReRaCOT, IReRaRetrieve, IReRaRetrieveRank
from .irera_utils import rp_at_k
from langProBe.benchmark import BenchmarkMeta
import subprocess

subprocess.run(["bash", "langProBe/IReRa/load_data.sh"], capture_output=True, text=True)

programs = [IReRaPredict(), IReRaCOT(), IReRaRetrieve(), IReRaRetrieveRank()]
benchmark = [BenchmarkMeta(IReRaBench, programs, rp_at_k)]

# making sure minibatch_size is not too large than validation set
for benchmark_indv in benchmark:
    for optimizer in benchmark_indv.optimizers:
        if optimizer.compile_args["minibatch_size"] > 30:
            optimizer.compile_args["minibatch_size"] = 30
