cd /home/kgajdamo/workspace/repos/ao-kgajdamo/benchmarks/inference

# Default (torch_compile_mode="default")
python bench_float8_inference.py

# With specific compile mode
python bench_float8_inference.py --torch_compile_mode default
python bench_float8_inference.py --torch_compile_mode reduce-overhead
python bench_float8_inference.py --torch_compile_mode max-autotune