# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Comprehensive documentation
- Performance optimizations
- Additional kernel types (more Matern variants, spectral kernels)

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

[Unreleased]: https://github.com/SimLej18/kernax-ml/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/SimLej18/kernax-ml/compare/v0.1.5-alpha...v0.2.0
[0.1.5-alpha]: https://github.com/SimLej18/kernax-ml/compare/v0.1.4-alpha...v0.1.5-alpha
[0.1.4-alpha]: https://github.com/SimLej18/kernax-ml/compare/v0.1.3-alpha...v0.1.4-alpha
[0.1.3-alpha]: https://github.com/SimLej18/kernax-ml/compare/v0.1.2-alpha...v0.1.3-alpha
[0.1.2-alpha]: https://github.com/SimLej18/kernax-ml/compare/v0.1.0...v0.1.2-alpha
[0.1.0]: https://github.com/SimLej18/kernax-ml/releases/tag/v0.1.0
