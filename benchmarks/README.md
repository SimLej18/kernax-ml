# Benchmarks

Simple and intuitive benchmarking suite for kernax-ml kernels.

## Structure

```
benchmarks/
├── input_generators.py      # Reusable input generation functions
├── conftest.py              # Pytest configuration (--bench-rounds option)
├── base_kernels/            # Benchmarks for base kernels
│   └── test_benchmark_se_kernel.py  # SEKernel benchmarks
├── comparison/              # Cross-library comparison benchmarks
│   └── test_compare_se_kernel.py    # Compare kernax vs sklearn/GPyTorch/GPJax
└── README.md                # This file
```

## Quick Start

Run all benchmarks with default settings (20 rounds):

```bash
# Run kernax performance benchmarks
make benchmarks

# Run cross-library comparison benchmarks
make benchmarks-compare
```

## Customization

### Control number of rounds

```bash
# Run with 50 rounds for more stable results
pytest benchmarks/ --benchmark-only --bench-rounds=50

# Quick benchmark with fewer rounds for faster feedback
pytest benchmarks/ --benchmark-only --bench-rounds=5
```

### Run specific benchmarks

```bash
# Run only 1D benchmarks
pytest benchmarks/ --benchmark-only -k "1d"

# Run only random input benchmarks
pytest benchmarks/ --benchmark-only -k "random"

# Run only base kernel benchmarks
pytest benchmarks/base_kernels/ --benchmark-only

# Run specific test
pytest benchmarks/base_kernels/test_benchmark_se_kernel.py::BenchmarkSEKernel::test_benchmark_2d_random --benchmark-only
```

### Save results

```bash
# Save to JSON
pytest benchmarks/ --benchmark-only --benchmark-json=benchmarks/out/results.json

# Save and compare with previous runs
pytest benchmarks/ --benchmark-only --benchmark-save=run_001
pytest-benchmark compare benchmarks/out/*
```

### Verbose output

```bash
# Show detailed benchmark stats
pytest benchmarks/ --benchmark-only -v

# Show even more details
pytest benchmarks/ --benchmark-only -vv
```

## Structure

Each benchmark test follows a simple pattern:

1. **Setup phase** (called before each round):
   - Instantiate the kernel
   - Generate appropriate input data
   - Warm up JIT compilation with exact input dimensions
   - Return kernel and inputs

2. **Execution phase** (timed):
   - Run kernel computation
   - Synchronize with `block_until_ready()`

This ensures JIT compilation overhead is excluded from timing measurements.

## Benchmark Coverage

### SEKernel

- **1D**: Regular grid (10k points), random inputs (10k points)
- **2D**: Regular grid (10k points), random inputs (10k points), missing values (10k points, 25% NaN)
- **Batched 1D**: 100 batches × 100 points with shared hyperparameters, 100 batches × 100 points with distinct hyperparameters
- **Batched 2D**: 100 batches × 100 points with shared hyperparameters, 100 batches × 100 points with distinct hyperparameters

### Cross-Library Comparisons

Comparison benchmarks for SE/RBF kernel across different libraries:
- **kernax**: SEKernel
- **scikit-learn**: RBF
- **GPyTorch**: RBFKernel
- **GPJax**: RBF

Scenarios tested:
- **1D Regular Grid**: 10k points on regular grid
- **1D Random**: 10k random points
- **2D Regular Grid**: 10k points (100×100 grid)
- **2D Random**: 10k random points
- **2D Missing Values**: 10k random points with 25% NaN values

Run with:
```bash
# Run all comparison benchmarks
make benchmarks-compare

# Compare specific libraries only
pytest benchmarks/comparison/ --benchmark-only -k "kernax or sklearn"

# Save results for later comparison
pytest benchmarks/comparison/ --benchmark-only --benchmark-autosave
```

## Adding New Benchmarks

To add a benchmark for a new kernel:

1. Create a new test file in the appropriate directory (e.g., `base_kernels/test_benchmark_my_kernel.py`)
2. Import input generators from `benchmarks.input_generators`
3. Follow the pattern: instantiate kernel, generate inputs, warmup, and benchmark
4. Use `request.config.getoption("--bench-rounds")` to get rounds from command line

Example:

```python
import jax.random as jr
from kernax import MyKernel
from benchmarks.input_generators import generate_random_inputs

class BenchmarkMyKernel:
    @classmethod
    def setup_class(cls):
        """Initialize PRNG key for the class."""
        cls.key = jr.PRNGKey(42)

    def test_benchmark_my_case(self, benchmark, request):
        """Description of what this benchmarks."""
        rounds = int(request.config.getoption("--bench-rounds"))

        def setup():
            # Split key to get new data each round
            self.key, subkey = jr.split(self.key)

            # Instantiate kernel
            kernel = MyKernel(param=1.0)

            # Generate inputs using helper
            x = generate_random_inputs(subkey, n_points=1000, n_dims=1, min_val=0, max_val=10)

            # Warmup JIT
            kernel(x, x).block_until_ready()

            return (kernel, x, x), {}

        def run_kernel(kernel, x1, x2):
            result = kernel(x1, x2)
            result.block_until_ready()

        benchmark.pedantic(run_kernel, setup=setup, rounds=rounds, iterations=1)
```

## Tips

- Start with fewer rounds for quick feedback during development
- Use more rounds for stable, production-quality benchmarks
- JIT warmup in `setup()` ensures measurements reflect actual execution time
- `block_until_ready()` ensures proper JAX synchronization for accurate timing