# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Comprehensive documentation
- Additional kernel types (more Matern variants, spectral kernels)

## [0.4.0-alpha] - 2025-01-31

### Added
- **Parameter Transform System:**
  - Global configuration system (`kernax.config`) for parameter transformation modes
  - Three transformation modes: `identity` (no transform), `exp` (log-exp trick), `softplus` (numerically stable)
  - Config locking mechanism prevents changes after kernel instantiation (ensures JIT compatibility)
  - `unsafe_reset()` method for testing and development environments
- **Transforms Module** (`kernax/transforms.py`):
  - `to_unconstrained()` and `to_constrained()` functions for bijective transformations
  - Centralizes transformation logic for cleaner kernel implementations
  - Supports positivity constraints for hyperparameter optimization
- **New Kernels:**
  - `SigmoidKernel`: Hyperbolic tangent kernel `tanh(α⟨x, x'⟩ + c)` with alpha parameter transformation
- **Test Suite:**
  - Comprehensive config system tests (14 tests, 100% coverage on config.py)
  - Transform module tests (10 tests)
  - SigmoidKernel tests (15 tests, 100% coverage on Sigmoid.py)
  - All SigmoidKernel tests integrated into `test_base_kernels.py`

### Changed
- **Code Organization:**
  - Reorganized kernels into family-specific directories:
    - `kernax/stationary/`: SE, Matern (1/2, 3/2, 5/2), Periodic, RationalQuadratic
    - `kernax/dotproduct/`: Linear, Polynomial, Sigmoid
    - `kernax/other/`: Constant, WhiteNoise
  - Updated all `__init__.py` files for new directory structure
- **Kernel Architecture:**
  - All stationary kernels now inherit from `StaticStationaryKernel` base class
  - All dot-product kernels now inherit from `StaticDotProductKernel` base class
  - Kernels use `cls.distance_func()` for distance computation (euclidean, squared_euclidean, or dot_product)
- **Parameter System:**
  - All kernels updated with parameter validation using `eqx.error_if`
  - Positive parameters stored in unconstrained space with property-based access
  - Parameters automatically transformed based on global config at instantiation time
  - Kernels updated: SEKernel, Matern12/32/52, Periodic, RationalQuadratic, Polynomial, Sigmoid
- **LinearKernel Special Case:**
  - LinearKernel parameters NOT transformed (variances can be zero, incompatible with log-based transforms)
  - Changed validation to `>= 0` instead of `> 0` for variance parameters
- **AbstractKernel:**
  - Added `__init__()` method to mark kernel instantiation for config locking

### Fixed
- **Transform Compatibility:**
  - Fixed LinearKernel to work with all transform modes by not transforming parameters
  - All 12 LinearKernel tests now pass
- **Test Organization:**
  - Removed separate `test_sigmoid_kernel.py`, integrated into `test_base_kernels.py`
  - Added `sample_1d_data` fixture to `test_base_kernels.py`

### Technical Details
- **Config System:** Thread-safe configuration with lock mechanism after first kernel creation
- **Property Pattern:** Kernels store `_unconstrained_param` and expose `param` property for constrained access
- **JIT Compatibility:** All transformations occur at instantiation time, not during JIT tracing
- **Backward Compatibility:** Default `identity` transform maintains existing behavior

## [0.3.1-alpha] - 2025-01-30

### Fixed
- **BatchKernel:** Fixed critical bug with `batch_in_axes=None` and `batch_over_inputs=False`
  - Replaced `cond` JAX (which traces both branches) with `if/else` Python for static conditions
  - Now correctly handles case where all hyperparameters and inputs are shared
  - Resolves `ValueError: vmap must have at least one non-None value in in_axes`
- **BlockKernel:** Fixed same issue with `block_in_axes=None` and `block_over_inputs=False`
  - Added conditional logic to avoid vmap when all axes are None
  - Simplified tile operation for creating block matrices with identical blocks
  - Correctly handles shared hyperparameters and shared inputs
- **BlockDiagKernel:** Inherits fix from BatchKernel, now works correctly with shared parameters

### Added
- **Test Coverage:** 4 new tests for edge cases with shared hyperparameters and inputs
  - `TestBatchKernel::test_shared_hyperparameters_shared_inputs`
  - `TestBlockKernel::test_shared_hyperparameters_shared_inputs`
  - `TestBlockDiagKernel::test_shared_hyperparameters_shared_inputs`
  - `TestWrapperCombinations::test_batch_block_commutativity` - verifies that `BlockKernel(BatchKernel(x))` equals `BatchKernel(BlockKernel(x))`

### Changed
- **Code Organization:** Improved internal structure of wrapper kernels for better maintainability
- **Test Suite:** Increased wrapper kernel test coverage (4 new tests, all passing)

## [0.3.0-alpha] - 2025-01-28

### Added
- **Benchmark Infrastructure:**
  - Comprehensive benchmarking suite for base kernels (`benchmarks/base_kernels/`)
  - Cross-library comparison benchmarks (kernax vs sklearn, GPyTorch, GPJax)
  - Parametrized test structure for comparing implementations
  - Support for `--bench-rounds` configuration
  - Automatic benchmark saving and comparison via pytest-benchmark
  - Input generators module for reusable benchmark data generation
- **Makefile Commands:**
  - `make benchmarks` for performance benchmarks with grouping and comparison
  - `make benchmarks-compare` for cross-library comparisons with autosave

### Changed
- **Performance:**
  - Optimized distance computation in SEKernel for better performance
- **CI/CD:**
  - Updated GitHub Actions linting workflow to use `ruff` instead of black/flake8/isort
  - Aligned CI linting with `make lint` commands

## [0.2.1-alpha] - 2025-01-27

### Added
- **Testing Infrastructure:**
  - Comprehensive tests for `BlockKernel` and `BlockDiagKernel` wrapper kernels (14 new tests)
  - Enhanced composition tests with `ExpKernel` and `LogKernel` coverage
  - String representation tests for all base kernels
  - Mathematical property tests (associativity, distributivity) for kernel compositions
  - Test coverage increased from 88% to 94% (231 passing tests)
- **Makefile Commands:**
  - `make test-allure` command for generating Allure one-file HTML reports
  - Centralized test output directory (`tests/out/`) for all test artifacts

### Changed
- **Import Structure:**
  - Standardized all internal imports to use relative imports (`from .AbstractKernel import`)
  - Fixes circular import issues and improves type checking
- **Test Organization:**
  - Merged `test_composite_kernels.py` into `test_kernel_compositions.py`
  - Removed redundant tests while preserving unique functionality
  - All composition tests now in single file (50 comprehensive tests)
- **Test Configuration:**
  - pytest outputs now save to `tests/out/htmlcov/` instead of root directory
  - Allure results and reports now save to `tests/out/`
- **Code Quality:**
  - Fixed all mypy linting errors (86 errors reduced to 0)
  - Added proper type annotations with `ClassVar` and `Optional`
  - Added `# type: ignore` comments for JAX-specific type issues

### Fixed
- **GitHub Actions:** Unified Python version to 3.12 for consistency across CI/CD
- **Documentation:**
  - Corrected Python requirement from 3.14 to 3.12 in README.md
  - Updated CLAUDE.md with BlockKernel/BlockDiagKernel documentation
  - Added testing guidelines and code quality standards to CLAUDE.md

### Removed
- **Deprecated Files:**
  - `setup.py` (replaced by `pyproject.toml` build system)
  - `test_composite_kernels.py` (merged into `test_kernel_compositions.py`)

### Development
- `.gitignore` updated to exclude `tests/out/` directory
- All test outputs (coverage, allure reports) now centralized in `tests/out/`

## [0.2.0-alpha] - 2025-01-26

### Added
- **New Kernels:**
  - `PolynomialKernel`: Polynomial kernel for non-stationary modeling
  - `RBFKernel`: Convenient alias for `SEKernel` (common in ML literature)
  - `WhiteNoiseKernel`: Convenient shortcut for `DiagKernel(ConstantKernel(value))`
- **Testing Infrastructure:**
  - Allure test reporting integration for comprehensive test analytics
  - Cross-library validation tests comparing against scikit-learn, GPyTorch, and GPJax
  - Parameterized tests across all base kernels for thorough hyperparameter coverage
  - NaN handling tests for robust behavior with missing data
  - Dimension handling tests for automatic shape inference
- **Test Configuration:**
  - `allurerc.json` for Allure test reporting configuration
  - `sample_batched_data` fixture in conftest.py

### Changed
- **RationalQuadraticKernel:** Removed `variance` hyperparameter for consistency with other implementations
- **Test Suite Modernization:**
  - All tests now use Allure decorators (`@allure.title`, `@allure.description`)
  - Extensive parameterization for better test coverage
  - Unified test structure across all kernel types

### Fixed
- **ARDKernel:** Fixed critical issue preventing proper operation - all wrapper kernel tests now passing
- **WrapperKernels.py:** Multiple fixes including:
  - JAX JIT compilation errors
  - Missing numpy import
  - Equinox Module field assignment issues
- **PolynomialKernel:** Fixed import issue in `__init__.py`

### Enhanced
- **Test Coverage:** Increased from basic coverage to comprehensive testing with:
  - ~170+ parameterized test cases for base kernels
  - Mathematical property validation (symmetry, positive semi-definiteness, etc.)
  - Cross-library consistency checks
  - Edge case handling (NaN, different dimensions)
- **Test Organization:**
  - Integrated batched operations tests into `test_base_kernels.py`
  - Modernized `test_composite_kernels.py` with 38 passing tests
  - Added algebraic property tests (associativity, distributivity)
  - Unified `TestDiagKernel` class (removed duplicate)

### Removed
- **Deprecated Tests:** `test_batched_operations.py` (functionality migrated to `test_base_kernels.py`)

### Development
- Updated `requirements-dev.txt` with Allure dependencies

## [0.1.5-alpha] - 2025-01-22

### Added
- BlockDiagKernel based on BatchedKernel.

### Fixed
- Output shape of BlockKernel, which wasn't flat

## [0.1.4-alpha] - 2025-01-21

### Fixed
- Missing import in `WrapperKernels.py`

## [0.1.3-alpha] - 2025-01-21

### Added
- New kernel: `BlockKernel` for block-structured covariance matrices

## [0.1.2-alpha] - 2024-12-02

### Added
- Readable `__str__()` methods for all kernel types with improved formatting
  - Scalar parameters display as `KernelName(param=value)`
  - Array parameters display as `[mean ± std]_shape` with unicode subscripts
  - Operator kernels display naturally (e.g., `K1 + K2`, `K1 * K2`)
  - Wrapper kernels use functional notation (e.g., `Exp(K)`, `Log(K)`)
  - Smart parentheses handling for complex expressions
- Utility functions: `format_jax_array()`, `to_subscript()`, `to_superscript()` in `kernax/utils.py`
- Demo notebook `tests/test_kernel_formatting.ipynb` showcasing kernel formatting capabilities

### Changed
- Downgraded Python requirement to `>=3.12` (from 3.14) for broader compatibility
- Downgraded JAX requirement to `>=0.6.2` (from 0.8.0) for broader compatibility
- Updated error messages in `AbstractKernel.__call__()` for clearer diagnostics
- Translated all French comments and documentation to English throughout the codebase

### Fixed
- Improved kernel string representation for better debugging and logging

## [0.1.0] - 2024-11-20

### Added
- Initial release of Kernax
- Core kernel implementations:
  - RBF (Radial Basis Function) kernel
  - Linear kernel
  - Matern family (1/2, 3/2, 5/2)
  - Periodic kernel
  - Rational Quadratic kernel
  - Constant kernel
  - SEMagma kernel
- Composite kernels (Sum, Product)
- Wrapper kernels (Diag, Exp, Log, Neg)
- Automatic dimension handling (scalar, vector, matrix, batched)
- NaN-aware computations for padded data
- Support for distinct hyperparameters per batch
- JAX PyTree integration for gradient computation
- Operator overloading for kernel composition (+, *, -)
- Basic test suite
- Documentation (README, CLAUDE.md, CONTRIBUTING)
- Package configuration (pyproject.toml, setup.py)

### Dependencies
- JAX >= 0.8.0
- JAXlib >= 0.8.0
- Equinox >= 0.11.0

[Unreleased]: https://github.com/SimLej18/kernax-ml/compare/v0.4.0-alpha...HEAD
[0.4.0-alpha]: https://github.com/SimLej18/kernax-ml/compare/v0.3.1-alpha...v0.4.0-alpha
[0.3.1-alpha]: https://github.com/SimLej18/kernax-ml/compare/v0.3.0-alpha...v0.3.1-alpha
[0.3.0-alpha]: https://github.com/SimLej18/kernax-ml/compare/v0.2.1-alpha...v0.3.0-alpha
[0.2.1-alpha]: https://github.com/SimLej18/kernax-ml/compare/v0.2.0-alpha...v0.2.1-alpha
[0.2.0-alpha]: https://github.com/SimLej18/kernax-ml/compare/v0.1.5-alpha...v0.2.0-alpha
[0.1.5-alpha]: https://github.com/SimLej18/kernax-ml/compare/v0.1.4-alpha...v0.1.5-alpha
[0.1.4-alpha]: https://github.com/SimLej18/kernax-ml/compare/v0.1.3-alpha...v0.1.4-alpha
[0.1.3-alpha]: https://github.com/SimLej18/kernax-ml/compare/v0.1.2-alpha...v0.1.3-alpha
[0.1.2-alpha]: https://github.com/SimLej18/kernax-ml/compare/v0.1.0...v0.1.2-alpha
[0.1.0]: https://github.com/SimLej18/kernax-ml/releases/tag/v0.1.0
