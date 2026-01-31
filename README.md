# Kernax - The blazing-fast kernel library that scales 🚀

Kernax is a Python package providing **efficient mathematical kernel implementations** for *probabilistic machine 
learning models*, built with the **JAX framework**.

Kernels are critical elements of probabilistic models, used in many inner-most loops to compute giant matrices and 
optimise numerous hyper-parameters. Therefore, this library emphasise on efficient, modular and scalable kernel 
implementations, with the following features:


- **JIT-compiled** computations for fast execution on CPU, GPU and TPU
- Kernels structured as **Equinox Modules** (aka *PyTrees*) which means...
  - They can be sent to (jitted) functions **as parameters**
  - Their **hyper-parameters can be optimised via autodiff**
  - They can **be vectorised-on** with `vmap`
- **Composable kernels** through operator overloading (`+`, `*`, `-`)
- **Kernel wrappers** to scale to higher dimensions (batch or block of covariance matrices)
- **NaN-aware computations** for working with padded/masked data

> **⚠️ Project Status**: Kernax is in early development. The API may change, and some features are still experimental.


## Installation

Install from PyPI:

```bash
pip install kernax-ml
```

Or clone the repository for development:

```bash
git clone https://github.com/SimLej18/kernax-ml
cd kernax-ml
```

**Requirements**:
- Python >= 3.12
- JAX >= 0.6.2

**Using Conda** (recommended):

```bash
conda create -n kernax-ml python=3.12
conda activate kernax-ml
pip install -e .
```

**Using pip**:

```bash
pip install -e .
```

## Quick Start

```python
import jax.numpy as jnp
from kernax import SEKernel, LinearKernel, DiagKernel, ExpKernel, BatchKernel, ARDKernel

# Create a simple Squared Exponential kernel
kernel = SEKernel(length_scale=1.0)

# Compute covariance between two points
x1 = jnp.array([1.0, 2.0])
x2 = jnp.array([1.5, 2.5])
cov = kernel(x1, x2)

# Compute covariance matrix for a set of points
X = jnp.array([[1.0], [2.0], [3.0]])
K = kernel(X, X)  # Returns 3x3 covariance matrix

# Compose kernels using operators
composite_kernel = SEKernel(length_scale=1.0) + DiagKernel(ExpKernel(0.1))  # SE + noise

# Use BatchKernel for distinct hyperparameters per batch
base_kernel = SEKernel(length_scale=1.0)
batched_kernel = BatchKernel(base_kernel, batch_size=10, batch_in_axes=0, batch_over_inputs=True)

# Use ARDKernel for Automatic Relevance Determination
length_scales = jnp.array([1.0, 2.0, 0.5])  # Different scale per dimension
ard_kernel = ARDKernel(SEKernel(length_scale=1.0), length_scales=length_scales)
```

## Available Kernels

### Base Kernels

- **`SEKernel`** (Squared Exponential, aka RBF or Gaussian)
  - Hyperparameters: `length_scale`

- **`LinearKernel`**
  - Hyperparameters: `variance_b`, `variance_v`, `offset_c`

- **`MaternKernel`** family
  - `Matern12Kernel` (ν=1/2, equivalent to Exponential)
  - `Matern32Kernel` (ν=3/2)
  - `Matern52Kernel` (ν=5/2)
  - Hyperparameters: `length_scale`

- **`PeriodicKernel`**
  - Hyperparameters: `length_scale`, `variance`, `period`

- **`RationalQuadraticKernel`**
  - Hyperparameters: `length_scale`, `variance`, `alpha`

- **`ConstantKernel`**
  - Hyperparameters: `value`

- **`PolynomialKernel`**
  - Hyperparameters: `degree`, `gamma`, `constant`

- **`SigmoidKernel`** (Hyperbolic Tangent)
  - Hyperparameters: `alpha`, `constant`

- **`WhiteNoiseKernel`**
  - Convenient shortcut for `DiagKernel(ConstantKernel(value))`
  - Hyperparameters: `value`

### Composite Kernels

- **`SumKernel`**: Adds two kernels (use `kernel1 + kernel2`)
- **`ProductKernel`**: Multiplies two kernels (use `kernel1 * kernel2`)

### Wrapper Kernels

Transform or modify kernel behavior:

- **`DiagKernel`**: Returns value only when inputs are equal (creates diagonal matrices)
- **`ExpKernel`**: Applies exponential to kernel output
- **`LogKernel`**: Applies logarithm to kernel output
- **`NegKernel`**: Negates kernel output (use `-kernel`)
- **`BatchKernel`**: Adds batch handling with distinct hyperparameters per batch
- **`BlockKernel`**: Constructs block covariance matrices for grouped data
- **`ActiveDimsKernel`**: Selects specific input dimensions before kernel computation
- **`ARDKernel`**: Applies Automatic Relevance Determination (different length scale per dimension)

## Architecture

Kernax is built on [Equinox](https://github.com/patrick-kidger/equinox), so they are compatible with every feature from JAX!

Each kernel uses a dual-class pattern to separate state and structure:

1. **Static Class** (e.g., `StaticSEKernel`): Contains JIT-compiled computation logic
2. **Instance Class** (e.g., `SEKernel`): Extends `eqx.Module`, holds hyperparameters

## Testing & Quality

Kernax maintains high code quality standards:

- **94% test coverage** with 231+ passing tests
- **Allure test reporting** for detailed test analytics
- **Cross-library validation** against scikit-learn, GPyTorch, and GPJax
- **Type checking** with mypy for enhanced code safety
- **Code formatting** with ruff (tabs, line length 100)

Run tests with:
```bash
make test           # Run all tests
make test-cov       # Run tests with coverage report
make test-allure    # Generate Allure HTML report
make lint           # Run type checking and linting
```

## Benchmarks

Kernax is designed for performance. You can run a benchmark comparison with other libraries with:
```bash
make benchmarks-compare
```

Our preliminary results show a significant speed-up over alternatives when JIT compilation is enabled:
```
------------------------------------------------- benchmark 'benchmarks/comparison/compare_se_kernel.py::Benchmark1DRandom::test_compare': 4 tests ------------------------------------------------
Name (time in ms)               Min                 Max                Mean            StdDev              Median               IQR            Outliers  OPS (mops/s)            Rounds  Iterations
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_compare[kernax]        12.0889 (1.0)       14.9765 (1.0)       12.6690 (1.0)      0.6474 (1.0)       12.4472 (1.0)      0.5749 (1.29)          1;1   78,932.9054 (1.0)          20           1
test_compare[gpytorch]      43.5577 (3.60)      54.3097 (3.63)      44.5481 (3.52)     2.3159 (3.58)      43.9814 (3.53)     0.4448 (1.0)           1;1   22,447.6602 (0.28)         20           1
test_compare[gpjax]         67.3657 (5.57)      73.9067 (4.93)      68.8340 (5.43)     1.4448 (2.23)      68.5019 (5.50)     1.3212 (2.97)          3;1   14,527.6964 (0.18)         20           1
test_compare[sklearn]      328.1409 (27.14)    367.2989 (24.53)    334.7924 (26.43)    8.2573 (12.75)    332.5784 (26.72)    4.2589 (9.57)          1;1    2,986.9256 (0.04)         20           1
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

---------------------------------------------- benchmark 'benchmarks/comparison/compare_se_kernel.py::Benchmark1DRegularGrid::test_compare': 4 tests -----------------------------------------------
Name (time in ms)               Min                 Max                Mean             StdDev              Median               IQR            Outliers  OPS (mops/s)            Rounds  Iterations
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_compare[kernax]        11.8415 (1.0)       13.3571 (1.0)       12.5704 (1.0)       0.4358 (1.0)       12.5185 (1.0)      0.6904 (2.23)          8;0   79,551.8567 (1.0)          20           1
test_compare[gpytorch]      43.6603 (3.69)      55.0724 (4.12)      44.5337 (3.54)      2.4908 (5.72)      43.9668 (3.51)     0.3099 (1.0)           1;1   22,454.9091 (0.28)         20           1
test_compare[gpjax]         67.2976 (5.68)     119.1254 (8.92)      70.9630 (5.65)     11.3640 (26.08)     68.3379 (5.46)     0.8114 (2.62)          1;2   14,091.8552 (0.18)         20           1
test_compare[sklearn]      297.8652 (25.15)    316.3752 (23.69)    302.3710 (24.05)     4.4811 (10.28)    300.7931 (24.03)    3.5605 (11.49)         4;2    3,307.1952 (0.04)         20           1
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

---------------------------------------------- benchmark 'benchmarks/comparison/compare_se_kernel.py::Benchmark2DMissingValues::test_compare': 4 tests ----------------------------------------------
Name (time in ms)               Min                 Max                Mean             StdDev              Median                IQR            Outliers  OPS (mops/s)            Rounds  Iterations
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_compare[kernax]        11.9619 (1.0)       13.9954 (1.0)       12.7085 (1.0)       0.5785 (1.0)       12.4387 (1.0)       0.8605 (1.0)           5;0   78,687.2477 (1.0)          20           1
test_compare[gpytorch]      25.9657 (2.17)      30.2475 (2.16)      27.1899 (2.14)      1.3834 (2.39)      26.6297 (2.14)      1.4433 (1.68)          4;2   36,778.4048 (0.47)         20           1
test_compare[gpjax]         55.1528 (4.61)     136.3582 (9.74)     113.5941 (8.94)     26.1170 (45.15)    125.4449 (10.09)    16.2559 (18.89)         3;3    8,803.2728 (0.11)         20           1
test_compare[sklearn]      213.0630 (17.81)    272.6552 (19.48)    230.3581 (18.13)    15.2409 (26.35)    224.1107 (18.02)    14.8310 (17.23)         6;1    4,341.0674 (0.06)         20           1
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

------------------------------------------------- benchmark 'benchmarks/comparison/compare_se_kernel.py::Benchmark2DRandom::test_compare': 4 tests ------------------------------------------------
Name (time in ms)               Min                 Max                Mean            StdDev              Median               IQR            Outliers  OPS (mops/s)            Rounds  Iterations
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_compare[kernax]        13.9697 (1.0)       15.5467 (1.0)       14.5331 (1.0)      0.4723 (1.0)       14.3701 (1.0)      0.8122 (1.57)          8;0   68,808.3748 (1.0)          20           1
test_compare[gpytorch]      43.6454 (3.12)      50.5380 (3.25)      44.5098 (3.06)     1.4877 (3.15)      44.1243 (3.07)     0.5169 (1.0)           1;2   22,466.9466 (0.33)         20           1
test_compare[gpjax]         94.1932 (6.74)     103.3563 (6.65)      97.6833 (6.72)     2.1244 (4.50)      97.8704 (6.81)     3.0956 (5.99)          4;0   10,237.1672 (0.15)         20           1
test_compare[sklearn]      408.6985 (29.26)    440.7834 (28.35)    417.1601 (28.70)    8.0450 (17.03)    413.5329 (28.78)    9.0707 (17.55)         5;1    2,397.1614 (0.03)         20           1
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

---------------------------------------------- benchmark 'benchmarks/comparison/compare_se_kernel.py::Benchmark2DRegularGrid::test_compare': 4 tests ----------------------------------------------
Name (time in ms)               Min                 Max                Mean            StdDev              Median               IQR            Outliers  OPS (mops/s)            Rounds  Iterations
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_compare[kernax]        13.9134 (1.0)       16.3335 (1.0)       14.5133 (1.0)      0.7465 (1.0)       14.1642 (1.0)      0.8484 (1.0)           3;2   68,902.5346 (1.0)          20           1
test_compare[gpytorch]      43.9418 (3.16)      51.8444 (3.17)      45.7195 (3.15)     1.8991 (2.54)      45.4059 (3.21)     1.9106 (2.25)          2;2   21,872.5192 (0.32)         20           1
test_compare[gpjax]         93.1488 (6.69)     130.6572 (8.00)     102.3413 (7.05)     8.1851 (10.96)    101.9908 (7.20)     9.7936 (11.54)         2;1    9,771.2271 (0.14)         20           1
test_compare[sklearn]      381.4240 (27.41)    405.2938 (24.81)    387.4473 (26.70)    6.4979 (8.70)     384.8169 (27.17)    9.0655 (10.69)         2;0    2,580.9964 (0.04)         20           1
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```

## Development Status

Check the [changelog](CHANGELOG.md) for details.

### ✅ Completed

- Core kernel implementations (SE, Linear, Matern, Periodic, Sigmoid, etc.)
- Kernel composition via operators
- Equinox Module integration
- NaN-aware computations
- BatchKernel wrapper with distinct/shared hyper-parameters
- ARDKernel wrapper using input scaling
- ActiveDimsKernel wrapper for dimension selection
- BlockKernel for block-matrix covariances
- StationaryKernel and DotProductKernel base classes with proper inheritance
- Parameter transform system (identity, exp, softplus) for optimization stability
- Parameter positivity constraints with config-based transformation
- Comprehensive test suite (94% coverage)
- Benchmark architecture
- PyPI package distribution

### 🚧 In Progress / Planned

- Add computation engines for special cases (diagonal-only, etc.)
- Parameter freezing for optimisation
- Comprehensive benchmarks with multiple kernels and input scenarios
- Expanded documentation and tutorials

## Contributing

This project is in early development. Contributions, bug reports, and feature requests are welcome!

## Related Projects

Kernax is developed alongside [MagmaClust](https://github.com/SimLej18/MagmaClustPy), a clustering and Gaussian Process library.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

[Citation information to be added]
