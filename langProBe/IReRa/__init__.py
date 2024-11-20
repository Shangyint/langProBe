from .irera_data import IReRaBench
from .irera_program import IReRaPredict, IReRaCOT, IReRaRetrieve, IReRaRetrieveRank
from .irera_utils import rp_at_k
from langProBe.benchmark import BenchmarkMeta
import subprocess

subprocess.run(
    ["bash", "langProBe/IReRa/load_data.sh"], capture_output=True, text=True
)

# programs = [IReRaPredict(), IReRaCOT(), IReRaRetrieve(), IReRaRetrieveRank()]
programs = [IReRaPredict()]
benchmark = [BenchmarkMeta(IReRaBench, programs, rp_at_k)]
