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

## [0.1.0] - 2025-11-20

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

[Unreleased]: https://github.com/SimLej18/kernax/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/SimLej18/kernax/releases/tag/v0.1.0