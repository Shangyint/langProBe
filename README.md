## Installation

```bash
pip install git+https://github.com/stanfordnlp/dspy.git
```
## Proposed Structure
### Data

There are essentially two ways to load data - one is to put everything into `dspy/datasets`, which is probably okay for existing benchmarks. However, creating new datasets would require modifying the DSPy codebase, making this benchmark overly coupled with the library. The other way is to define a seperate methods for each task. See `langProBe/humaneval/humaneval_data.py` for an example.

### File structure for each benchmark
```
langProBe/
    bench_name/
        __init__.py
        bench_name_data.py
        bench_name_program.py
        bench_name_utils.py
    ...
```

#### `__init__.py`
This file should define `benchmark` to the class handling the dataset `BenchNameBench`, i.e.
```python
from .bench_name_data import BenchNameBench
benchmark = BenchNameBench
```

In addition, this file should define `programs` to all the programs available for this benchmark, i.e.
```python
from .bench_name_program import Program1, Program2, ...
programs = [Program1, Program2, ...]
```

This file should also define `metric` that applies to this benchmark, i.e.
```python
metric = lambda gold_example, pred: gold_example.answer == pred.answer
```

#### `bench_name_data.py`
This file defines the data used for this benchmark. It should download the data, preprocess it, and create a `Benchmark` subclass called `BenchNameBench`. 

#### `bench_name_program.py`
This file defines DSPy programs for this benchmark. Each program should be a subclass of `dspy.Module`. For simplicity, we require each program to be initialized directly, i.e. `Program()`.

#### `bench_name_utils.py`
This file defines utility functions for this benchmark.

## TODO architecture-level
[x] Top-level config to enable/disable benchmarks and programs
[ ] have summary/visualization/report after running all the experiments
[ ] Seperate between easier evaluations and harder evaluations. Easier ones require zero installation - runs on the go. For each benchmarks, have an easier way to output their size, easiness, time to evaluate etc.
[ ] top-level experiment scripts that involve vanilla DSPy programs and optimizers (we need training data here).

### Contributing
#### Formatting
For simplicity, we use `black` formatter with the following command:
```bash
black --fast langProBe/*.py langProBe/*/*.py
```
