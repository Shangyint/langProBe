from .irera_data import IReRaBench
from .irera_program import Infer, InferRetrieve, InferRetrieveRank
from .irera_utils import rp_at_k
from langProBe.benchmark import BenchmarkMeta
import subprocess

subprocess.run(
    ["bash", "langProBe/IReRa/load_data.sh"], capture_output=True, text=True
)

programs = [Infer(), InferRetrieve(), InferRetrieveRank()]
benchmark = [BenchmarkMeta(IReRaBench, programs, rp_at_k)]
