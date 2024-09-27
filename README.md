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
        bench_name_data.py
        bench_name_eval.py
        bench_name_program.py
        bench_name_utils.py
    ...
```

#### `bench_name_data.py`
This file defines the data used for this benchmark. It should download the data, preprocess it, and create a `Benchmark` subclass called `BenchNameBench`. 

#### `bench_name_program.py`
This file defines DSPy programs for this benchmark.
TODO(@shangyin): We can potentially requires this file to define an `Enum` class for program names, or even a dictornary that maps program names to their available features (including optimizers).

#### `bench_name_utils.py`
This file defines utility functions for this benchmark.

#### `bench_name_eval.py`
This file first defines optimizers for this benchmark. Then, it defines the evaluation script.

### Contributing
#### Formatting
For simplicity, we use `black` formatter with the following command:
```bash
black --fast **/*.py
```