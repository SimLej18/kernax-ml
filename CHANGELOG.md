# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Additional kernel types (more Matern variants, spectral kernels)

## [0.7.0-alpha] - 2026-08-15

### Added
- Multi-output Gaussian Process support (`kernax.multioutput`): `ICMKernel`, `LMCKernel`, `ConvolutionKernel`, `BlockDiagKernel`, `BlockMean`.
  - Common convention: `output_ids` (+ `output_ids2` for cross-covariances) is passed to `__call__`, not stored on the instance — a property of the data, not the model. Two input regimes per class: a shared grid across every output (`output_ids` omitted), or heterotopic data with arbitrary per-output sizes and ordering (`output_ids` given).
  - `ICMKernel`: Intrinsic Coregionalisation Model, `K(x1, x2) = B ⊗ k(x1, x2)` with `B = W Wᵀ` positive semi-definite by construction; `n_latent < n_outputs` gives a low-rank coregionalisation.
  - `LMCKernel`: Linear Model of Coregionalisation, a sum of independent `ICMKernel` components with arbitrary rank and kernel class per component.
  - `ConvolutionKernel`: convolution-process multi-output kernel (Alvarez & Lawrence, 2011), per-output marginal variance and bandwidth (optionally per-dimension via `ard=True`); ships `from_paper_parameters()` to build directly from the paper's own parametrisation. Uses its own engine — not compatible with the shared `engines` catalogue.
  - `BlockDiagKernel` / `BlockMean`: independent per-output kernel/mean with no cross-output correlation; per-output hyperparameters selected via `output_hps_in_axes` (built with `create_mask`), shared hyperparameters left untouched.
- `SumModule`/`ProductModule` now propagate `**kwargs` (e.g. `output_ids`) to both operands, so multi-output kernels compose correctly through `+`/`*` (e.g. `ICMKernel(...) + BlockDiagKernel(WhiteNoiseKernel(...), ...)` for correlated signal plus independent per-output noise).
- Example scripts: `docs/examples/multioutput_icm.py`, `multioutput_lmc.py`, `multioutput_convolution.py`.
- `tests/test_multioutput.py`: dedicated test suite for `kernax.multioutput`, covering both input regimes, cross-checks between the shared-grid and heterotopic code paths, and PSD/symmetry invariants.

### Changed *(breaking)*
- `ICMKernel`, `LMCKernel`, `ConvolutionKernel`: `feature_sizes`-based construction replaced by the `output_ids` convention described above.
- `BlockDiagKernel` moved from `kernax.wrappers` to `kernax.multioutput` and rebuilt on the `output_ids` convention. Old signature (`nb_blocks`, `block_in_axes`, `block_over_inputs`, fixed-size blocks only) is gone; new signature is `BlockDiagKernel(inner, n_outputs, output_hps_in_axes=None)`.
- `BatchModule` now built on top of the external `eqxbatch.Batched` rather than an in-house implementation; public behavior unchanged.

### Removed
- `BlockKernel`, `FeatureKernel`: the two-symmetric-hyperparameter-set use case they served is covered natively by `ConvolutionKernel`, with its own engine and the `output_ids` convention. No other code depended on either class.

### Fixed
- Missing `engine` field on `ConvolutionKernel`.
- `spectral_density` for `SumModule`.
- Missing imports in `Matern12Kernel`, `Matern32Kernel`, `Matern52Kernel`.

## [0.6.2-alpha] - 2026-06-30

### Added
- `spectral_density(w)` method on `AbstractStationaryKernel` (abstract) and all concrete implementations:
  - `SEKernel`: Gaussian spectral density `(2π)^(d/2) · l^d · exp(-l² ‖w‖² / 2)`
  - `Matern12Kernel`, `Matern32Kernel`, `Matern52Kernel`: closed-form Student-t spectral densities using the gamma function
  - `VarianceKernel`: returns `variance` (constant spectral density)
  - `ARDKernel`, `ActiveDimsModule`: propagate to inner kernel with appropriate frequency rescaling
  - `BatchModule`: vmaps over batched hyperparameters when applicable
  - `ProductModule`: supported when one operand is `VarianceKernel` (scales spectral density by variance)
  - `SumModule`, `FeatureKernel`, `PeriodicKernel`, `RationalQuadraticKernel`, `WhiteNoiseKernel`: raise `NotImplementedError` (no closed-form or not yet implemented)

## [0.6.1-alpha] - 2026-04-23

### Fixed
- Bug in `create_mask` when used with doubly-nested `BatchModule`: `_recurse` now correctly traverses inner/left/right sub-modules only, avoiding spurious replacements.

## [0.6.0-alpha] - 2026-04-22

### Added
- `InputSpecificParamModule`: wrapper giving each input its own set of HPs; designed for per-point noise with `WhiteNoiseKernel`.
- `MaskedNaNEngine`: new engine implementing NaN masking directly in the kernel matrix computation.
- Sphinx + ReadTheDocs documentation with full structure and getting-started content.

### Changed *(breaking)*
- JIT no longer applied internally; wrap kernels at the top level with `eqx.filter_jit(kernel)`.
- Architecture rewrite: abstract-final pattern with per-HP customizable parametrisation; `Static` classes removed.

### Fixed
- Bug in `WhiteNoiseKernel` computation.
- Python 3.10 compatibility: removed multi-line f-strings.

### Removed
- Cross-library comparison benchmarks (moved to KernelArena).

## [0.5.5-alpha] - 2026-03-16

### Added
- `AffineKernel`: new dot-product kernel (`slope · x + intercept`), split off from `LinearKernel`.
- `create_mask(module, **kwargs)` utility to build pytree masks on any `AbstractModule`; useful for freezing params or building custom `batch_in_axes`.

### Changed
- `LinearKernel` API updated to match the new dot-product kernel family conventions.

## [0.5.2-alpha] - 2026-02-25

### Added
- `sample_hps_from_uniform_priors(key, module, priors)` utility for random HP initialization from uniform bounds; supports constrained params, nested modules, and batched HPs.

## [0.5.1-alpha] - 2026-02-25

### Fixed
- `BatchModule.__call__` now correctly dispatches for mean functions (single-input) vs kernels (two-input).
- `replace()` now raises `ValueError` on immutable structural fields in `BatchModule`, `BlockKernel`, `BlockDiagKernel`, and `ActiveDimsModule` (was silently ignored).

## [0.5.0-alpha] - 2026-02-24

### Added
- `AbstractModule` base class unifying kernels and means (`replace()`, operators, `__str__()`).
- Mean functions: `ZeroMean`, `ConstantMean`, `LinearMean`, `AffineMean`.
- All operators and wrappers now work with means as well as kernels.

### Changed *(breaking)*
- Operator/wrapper classes renamed from `*Kernel` to `*Module` suffix (`SumKernel` → `SumModule`, etc.). Old names no longer exported.

## [0.4.4-alpha] - 2026-02-06

### Added
- `VarianceKernel`: scalar constant kernel for standalone variance in compositions.
- Immutable HP modification API: `replace(**kwargs)` on all kernel types, `modify_left/right()` on operators, `modify_inner()` on wrappers.

### Changed
- `__str__()` now shows constrained parameter values.
- Internal `_unconstrained_*` attributes renamed to `_raw_*` (non-breaking).
- `WhiteNoiseKernel` reimplemented on top of `VarianceKernel`.

## [0.4.3-alpha] - 2026-02-05

### Added
- Initial support for kernel modifications.

### Fixed
- Bug in `FeatureKernel` hyperparameter handling.

## [0.4.2-alpha] - 2026-02-03

### Added
- `FeatureKernel`: designed for use with `BlockKernel` when HPs vary across blocks.

### Changed *(breaking)*
- `BlockKernel` API: `block_in_axes` now expects a pytree (`0` for per-block HPs, `None` for shared).

## [0.4.1-alpha] - 2026-02-02

### Fixed
- `SafeRegularGridEngine`: incorrect `vmap` call in `check_constraints`.
- `SafeDiagonalEngine`: wrong attribute reference and missing `cross_cov_matrix` implementation.

### Changed
- `WhiteNoiseKernel` reimplemented using `SafeDiagonalEngine` (inherits from `ConstantKernel`).

### Removed
- `DiagKernel` — use `SafeDiagonalEngine` or `FastDiagonalEngine` instead.

## [0.4.0-alpha] - 2025-01-31

### Added
- Parameter transform system: `identity`, `exp`, `softplus` modes via `kernax.config`.
- `to_unconstrained()` / `to_constrained()` in `kernax/transforms.py`.
- `SigmoidKernel`: hyperbolic tangent kernel.

### Changed
- Kernels reorganized into `stationary/`, `dotproduct/`, `other/` subdirectories.
- All stationary/dot-product kernels use shared base classes and `cls.distance_func()`.
- Positive HPs now stored in raw space with property-based constrained access.

## [0.3.1-alpha] - 2025-01-30

### Fixed
- `BatchKernel`, `BlockKernel`, `BlockDiagKernel`: replaced JAX `cond` with Python `if/else` for static `batch_in_axes=None` / `batch_over_inputs=False` cases (resolved vmap error).

## [0.3.0-alpha] - 2025-01-28

### Added
- Benchmark infrastructure: base kernel suite, cross-library comparisons (sklearn, GPyTorch, GPJax), `make benchmarks` and `make benchmarks-compare` commands.

### Changed
- CI linting switched from black/flake8/isort to `ruff`.

## [0.2.1-alpha] - 2025-01-27

### Added
- Tests for `BlockKernel`, `BlockDiagKernel`, `ExpKernel`, `LogKernel`, and mathematical composition properties. Coverage: 88% → 94%.
- `make test-allure` command; test outputs centralized in `tests/out/`.

### Changed
- All internal imports standardized to relative imports.
- Fixed all mypy errors (86 → 0).

## [0.2.0-alpha] - 2025-01-26

### Added
- `PolynomialKernel`, `RBFKernel` alias, `WhiteNoiseKernel`.
- Allure test reporting; cross-library validation tests.

### Fixed
- `ARDKernel`, `WrapperKernels.py` (JIT errors, missing imports, field assignment).

### Changed
- `RationalQuadraticKernel`: removed `variance` parameter.

## [0.1.5-alpha] - 2025-01-22

### Added
- `BlockDiagKernel` based on `BatchKernel`.

### Fixed
- `BlockKernel` output shape (was not flat).

## [0.1.4-alpha] - 2025-01-21

### Fixed
- Missing import in `WrapperKernels.py`.

## [0.1.3-alpha] - 2025-01-21

### Added
- `BlockKernel` for block-structured covariance matrices.

## [0.1.2-alpha] - 2024-12-02

### Added
- `__str__()` for all kernel types with smart formatting (scalar params, array stats, operator notation).
- `format_jax_array()`, `to_subscript()`, `to_superscript()` utilities.

### Changed
- Python requirement lowered to `>=3.12`; JAX to `>=0.6.2`.

## [0.1.0] - 2024-11-20

### Added
- Initial release: SE, Linear, Matern (1/2, 3/2, 5/2), Periodic, RationalQuadratic, Constant, SEMagma kernels.
- Sum/Product composite kernels; Diag/Exp/Log/Neg wrappers.
- Automatic dimension handling, NaN-aware computations, JAX PyTree integration, operator overloading.

[Unreleased]: https://github.com/SimLej18/kernax-ml/compare/v0.7.0-alpha...HEAD
[0.7.0-alpha]: https://github.com/SimLej18/kernax-ml/compare/v0.6.2-alpha...v0.7.0-alpha
[0.6.2-alpha]: https://github.com/SimLej18/kernax-ml/compare/v0.6.1-alpha...v0.6.2-alpha
[0.6.1-alpha]: https://github.com/SimLej18/kernax-ml/compare/v0.6.0-alpha...v0.6.1-alpha
[0.6.0-alpha]: https://github.com/SimLej18/kernax-ml/compare/v0.5.5-alpha...v0.6.0-alpha
[0.5.5-alpha]: https://github.com/SimLej18/kernax-ml/compare/v0.5.4-alpha...v0.5.5-alpha
[0.5.4-alpha]: https://github.com/SimLej18/kernax-ml/compare/v0.5.2-alpha...v0.5.4-alpha
[0.5.2-alpha]: https://github.com/SimLej18/kernax-ml/compare/v0.5.1-alpha...v0.5.2-alpha
[0.5.1-alpha]: https://github.com/SimLej18/kernax-ml/compare/v0.5.0-alpha...v0.5.1-alpha
[0.5.0-alpha]: https://github.com/SimLej18/kernax-ml/compare/v0.4.4-alpha...v0.5.0-alpha
[0.4.4-alpha]: https://github.com/SimLej18/kernax-ml/compare/v0.4.3-alpha...v0.4.4-alpha
[0.4.3-alpha]: https://github.com/SimLej18/kernax-ml/compare/v0.4.2-alpha...v0.4.3-alpha
[0.4.2-alpha]: https://github.com/SimLej18/kernax-ml/compare/v0.4.1-alpha...v0.4.2-alpha
[0.4.1-alpha]: https://github.com/SimLej18/kernax-ml/compare/v0.4.0-alpha...v0.4.1-alpha
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