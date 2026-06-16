#!/bin/bash

cd /home/kgajdamo/workspace/repos/ao-kgajdamo

# Default configuration (eager, all shapes)
python benchmarks/float8/bench_padding.py

# Eager mode, explicit
python benchmarks/float8/bench_padding.py --compile=False

# With torch.compile, all shapes
python benchmarks/float8/bench_padding.py --compile=True

# Limit number of experiments (eager)
python benchmarks/float8/bench_padding.py --n_limit=1
python benchmarks/float8/bench_padding.py --n_limit=3
python benchmarks/float8/bench_padding.py --n_limit=5

# Limit number of experiments (compile)
python benchmarks/float8/bench_padding.py --compile=True --n_limit=1
python benchmarks/float8/bench_padding.py --compile=True --n_limit=3
python benchmarks/float8/bench_padding.py --compile=True --n_limit=5