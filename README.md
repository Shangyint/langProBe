## Installation

```bash
pip install git+https://github.com/stanfordnlp/dspy.git
```
## Proposed Structure
### Data

There are essentially two ways to load data - one is to put everything into `dspy/datasets`, which is probably okay for existing benchmarks. However, creating new datasets would require modifying the DSPy codebase, making this benchmark overly coupled with the library. The other way is to define a seperate methods for each task. See `langProBe/humaneval/humaneval_bench.py` for an example.

