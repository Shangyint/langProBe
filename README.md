## Installation

```bash
pip install -r requirements.txt
```

## Quick Usage
```bash
# example with using gpt-4o, with all non-agent datasets
mkdir evaluation_gpt4o
DSPY_CACHEDIR=evaluation_gpt4o/.dspy_cache python -m langProBe.evaluation --benchmark_set=nonagent --file_path=evaluation_gpt4o --lm=openai/gpt-4o
```

```bash
# example with using llama (change `lm_api_base` to your API provider)
mkdir evaluation_llama3170b
DSPY_CACHEDIR=evaluation_llama3170b/.dspy_cache python -m langProBe.evaluation --benchmark_set=nonagent --file_path=evaluation_llama3170b --lm=openai/meta-llama/Meta-Llama-3.1-70b-Instruct --lm_api_base=http://future-hgx-1:7410/v1
```

```bash
# parse the result and generate figures
python -m langProBe.analysis --file_path=evaluation_llama3170b
```

## Adding Benchmarks, Programs, Optimizers

Benchmarks and programs are defined by the `BenchmarkMeta` class. You can program definitions to existing `BenchmarkMeta`s or define your own `BenchmarkMeta`s.
Additionally, each `BenchmarkMeta` object also has an `optimizers` field, containing optimizer definitions. You can inspect `optimizers.py` to checkout how to define an optimizer and default optimizers in `DEFAULT_OPTIMIZERS`.



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
This file should define `benchmark: List[BenchmarkMeta]`.

Each `BenchmarkMeta` should have the following fields:
- `benchmark: type[Benchmark]` - the benchmark class
- `programs: List[type[dspy.Module]]` - the programs for this benchmark
- `metric` - the metric for this benchmark

#### `bench_name_data.py`
This file defines the data used for this benchmark. It should download the data, preprocess it, and create a `Benchmark` subclass called `BenchNameBench`. 

#### `bench_name_program.py`
This file defines DSPy programs for this benchmark. Each program should be a subclass of `dspy.Module`. For simplicity, we require each program to be initialized directly, i.e. `Program()`.

#### `bench_name_utils.py`
This file defines utility functions for this benchmark.

## TODO architecture-level
- [ ] easier way to specify what program/dataset/optimzier to run. Let's use a config system!
- [ ] add a missing mode for running experiments
- [ ] make names mandatory for optimizers

### Contributing
#### Formatting
For simplicity, we use `black` formatter with the following command:
```bash
black --fast langProBe/*.py langProBe/*/*.py
```
