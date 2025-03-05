# LangProBe

LangProBe (language program benchmark) measures the architectures and optimization strategies for language model programs. Using LangProBe, exploring relationships between cost and language model program performance becomes easy. For a detailed report, read our paper [LangProBe: a Language Programs Benchmark](https://arxiv.org/abs/2502.20315). We welcome contributions from all dimensions, including new programs, new (DSPy) optimizations, new datasets, as well as experiment data for new language models. 

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

## Adding Benchmarks, Programs, Optimizers

Benchmarks and programs are defined by the `BenchmarkMeta` class. You can program definitions to existing `BenchmarkMeta`s or define your own `BenchmarkMeta`s.
Additionally, each `BenchmarkMeta` object also has an `optimizers` field, containing optimizer definitions. You can inspect `optimizers.py` to checkout how to define an optimizer and default optimizers in `DEFAULT_OPTIMIZERS`. Currently, optimizers can only be evaluated with DSPy programs.



## LangProBe Structure

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
This file defines programs for this benchmark. There are a few requirements for customized programs:
1. `setup_lm(lm:str, api_key:str, api_base:str)` - your program should support setting up lm with `setup_lm` method. We recommend the LiteLLM library.
2. program should provide a callable interface (through `__call__` method). The arguments will be a dictionary (`**kwargs`) of input keys to their actual values. For example, a question-answering dataset may use the following input `program(question="what is 1 + 1?")`. The answer should be a **DotDict** with respect field. The same example could be `return DotDict({"answer": "2"})`.

#### `bench_name_utils.py`
This file defines utility functions for this benchmark.

### Contributing
#### Formatting
For simplicity, we use `black` formatter with the following command:
```bash
black --fast langProBe/*.py langProBe/*/*.py
```
