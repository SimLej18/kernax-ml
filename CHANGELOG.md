# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Comprehensive documentation
- Additional kernel types
- Performance optimizations
- More extensive test coverage

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

[Unreleased]: https://github.com/SimLej18/kernax-ml/compare/v0.1.2-alpha...HEAD
[0.1.2-alpha]: https://github.com/SimLej18/kernax-ml/compare/v0.1.0...v0.1.2-alpha
[0.1.0]: https://github.com/SimLej18/kernax-ml/releases/tag/v0.1.0