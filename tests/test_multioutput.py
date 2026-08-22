"""
Tests for kernax.multioutput: BlockDiagKernel, BlockMean, ICMKernel, LCMKernel,
ConvolutionKernel, and the shared `gather_by_output` helper.

Focus is on invariants that are easy to get wrong across the two input regimes these
classes share (shared-grid vs heterotopic `output_ids`), not exhaustive shape checks.
"""

import allure
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest

from kernax import (
	BlockDiagKernel,
	BlockMean,
	ConstantMean,
	ConvolutionKernel,
	ICMKernel,
	LCMKernel,
	Matern32Kernel,
	SEKernel,
	WhiteNoiseKernel,
)
from kernax.mask import create_mask
from kernax.multioutput._gather import gather_by_output
from kernax.parametrisations import NonTrainableParametrisation


def _min_eigval(K):
	K = 0.5 * (K + K.T)
	return jnp.min(jnp.linalg.eigvalsh(K))


class TestGatherByOutput:
	"""Tests for the private helper shared by BlockDiagKernel and BlockMean."""

	@allure.title("gather_by_output indexes per-output leaves, leaves shared leaves untouched")
	@allure.description(
		"A leaf marked with axis 0 must be indexed by output_ids; a leaf marked None "
		"must pass through unchanged, not broadcast or indexed."
	)
	def test_gather_per_output_and_shared_leaves(self):
		inner = {"per_output": jnp.array([10.0, 20.0, 30.0]), "shared": jnp.array(5.0)}
		in_axes = {"per_output": 0, "shared": None}
		output_ids = jnp.array([2, 0, 0, 1])

		gathered = gather_by_output(inner, in_axes, output_ids)

		assert jnp.allclose(gathered["per_output"], jnp.array([30.0, 10.0, 10.0, 20.0]))
		assert gathered["shared"] == 5.0


class TestBlockDiagKernel:
	"""Tests for BlockDiagKernel (kernax/multioutput/BlockDiagKernel.py)."""

	def _kernel(self):
		base = SEKernel(length_scale=1.0)
		kernel = BlockDiagKernel(
			base, n_outputs=3, output_hps_in_axes=create_mask(base, default=None, length_scale=0)
		)
		return kernel.replace(length_scale=jnp.array([0.5, 1.0, 2.0]))

	@allure.title("BlockDiagKernel shared grid: diagonal blocks use per-output HPs, off-diag zero")
	@allure.description(
		"Not just shapes: each diagonal block must equal the individual kernel with that "
		"output's own length_scale, and off-diagonal blocks must be exactly zero."
	)
	def test_shared_grid_blocks(self):
		kernel = self._kernel()
		x = jnp.linspace(0.0, 5.0, 4)[:, None]
		N = x.shape[0]
		K = kernel(x)

		assert K.shape == (3 * N, 3 * N)
		for i, ls in enumerate([0.5, 1.0, 2.0]):
			block = K[i * N:(i + 1) * N, i * N:(i + 1) * N]
			assert jnp.allclose(block, SEKernel(length_scale=ls)(x, x))
		assert jnp.allclose(K[:N, N:2 * N], 0.0)
		assert jnp.allclose(K[:N, 2 * N:], 0.0)

	@allure.title("BlockDiagKernel: shared-grid route matches heterotopic route on the same data")
	@allure.description(
		"output_ids=None is documented as equivalent to tiling x across outputs and passing "
		"explicit output_ids -- two separate code paths for the same result, must agree exactly."
	)
	def test_shared_grid_matches_heterotopic_equivalent(self):
		kernel = self._kernel()
		x = jnp.linspace(0.0, 5.0, 4)[:, None]
		N, P = x.shape[0], 3

		K_shared = kernel(x)
		output_ids = jnp.repeat(jnp.arange(P), N)
		x_tiled = jnp.tile(x, (P, 1))
		K_hetero = kernel(x_tiled, output_ids=output_ids, output_ids2=output_ids)

		assert jnp.allclose(K_shared, K_hetero)

	@allure.title("BlockDiagKernel heterotopic: shuffled, uneven-size output_ids")
	@allure.description(
		"Ground truth computed independently (per-pair loop with a fresh per-output kernel "
		"instance), not by re-deriving the implementation's own masking formula."
	)
	def test_heterotopic_shuffled_uneven_sizes(self):
		kernel = self._kernel()
		x = jnp.array([[0.1], [1.3], [0.4], [2.7], [1.9]])
		output_ids = jnp.array([1, 0, 2, 0, 1])
		length_scales = jnp.array([0.5, 1.0, 2.0])

		K = kernel(x, output_ids=output_ids)

		for i in range(5):
			for j in range(5):
				if output_ids[i] != output_ids[j]:
					assert K[i, j] == 0.0
				else:
					ref = SEKernel(length_scale=length_scales[output_ids[i]])(x[i], x[j])
					assert jnp.allclose(K[i, j], ref)

	@allure.title("BlockDiagKernel heterotopic cross-covariance, different point sets")
	@allure.description("x1 and x2 have different sizes and different output_ids arrays.")
	def test_heterotopic_cross_covariance_different_sizes(self):
		kernel = self._kernel()
		x1 = jnp.array([[0.1], [1.3], [0.4], [2.7], [1.9]])
		x2 = jnp.array([[0.2], [1.1], [0.9]])
		ids1 = jnp.array([1, 0, 2, 0, 1])
		ids2 = jnp.array([0, 2, 1])

		K = kernel(x1, x2, output_ids=ids1, output_ids2=ids2)

		assert K.shape == (5, 3)
		assert jnp.array_equal(K == 0.0, ids1[:, None] != ids2[None, :])

	@allure.title("BlockDiagKernel: validation errors")
	@allure.description("output_ids2 without output_ids, and mismatched output_ids length.")
	def test_validation_errors(self):
		kernel = self._kernel()
		x = jnp.linspace(0.0, 5.0, 4)[:, None]

		with pytest.raises(ValueError):
			kernel(x, output_ids2=jnp.arange(4))
		with pytest.raises(ValueError):
			kernel(x, output_ids=jnp.arange(3))

	@allure.title("BlockDiagKernel: n_outputs is structurally immutable")
	@allure.description("replace(n_outputs=...) must raise, per the explicit guard in the code.")
	def test_replace_n_outputs_raises(self):
		kernel = self._kernel()
		with pytest.raises(ValueError):
			kernel.replace(n_outputs=5)


class TestBlockMean:
	"""Tests for BlockMean (kernax/multioutput/BlockMean.py)."""

	def _mean(self):
		base = ConstantMean(0.0)
		mean = BlockMean(
			base, n_outputs=3, output_hps_in_axes=create_mask(base, default=None, constant=0)
		)
		return mean.replace(constant=jnp.array([-1.0, 0.0, 1.0]))

	@allure.title("BlockMean shared grid: output-major flatten order")
	@allure.description(
		"The flattened result must be ordered output-major (output 0's points first, ...), "
		"not point-major -- this is what aligns m with the rows of a same-shaped BlockDiagKernel."
	)
	def test_shared_grid_output_major_order(self):
		base = ConstantMean(0.0)
		mean = BlockMean(
			base, n_outputs=2, output_hps_in_axes=create_mask(base, default=None, constant=0)
		)
		mean = mean.replace(constant=jnp.array([10.0, 20.0]))
		x = jnp.array([[0.0], [1.0], [2.0]])

		m = mean(x)

		assert jnp.allclose(m, jnp.array([10.0, 10.0, 10.0, 20.0, 20.0, 20.0]))

	@allure.title("BlockMean: shared-grid route matches heterotopic route on the same data")
	def test_shared_grid_matches_heterotopic_equivalent(self):
		mean = self._mean()
		x = jnp.linspace(0.0, 5.0, 4)[:, None]
		N, P = x.shape[0], 3

		m_shared = mean(x)
		output_ids = jnp.repeat(jnp.arange(P), N)
		x_tiled = jnp.tile(x, (P, 1))
		m_hetero = mean(x_tiled, output_ids=output_ids)

		assert jnp.allclose(m_shared, m_hetero)

	@allure.title("BlockMean heterotopic: shuffled, uneven-size output_ids")
	def test_heterotopic_shuffled_uneven_sizes(self):
		mean = self._mean()
		x = jnp.array([[0.1], [1.3], [0.4], [2.7], [1.9]])
		output_ids = jnp.array([1, 0, 2, 0, 1])
		constants = jnp.array([-1.0, 0.0, 1.0])

		m = mean(x, output_ids=output_ids)

		assert jnp.allclose(m, constants[output_ids])

	@allure.title("BlockMean: n_outputs is structurally immutable")
	def test_replace_n_outputs_raises(self):
		mean = self._mean()
		with pytest.raises(ValueError):
			mean.replace(n_outputs=5)


class TestICMKernel:
	"""Tests for ICMKernel (kernax/multioutput/ICMKernel.py)."""

	def _kernel(self, key):
		W = jr.normal(key, (3, 2))
		return ICMKernel(SEKernel(length_scale=1.0), n_outputs=3, n_latent=2).replace(W=W), W

	@allure.title("ICMKernel shared grid: dense result equals manual Kronecker product")
	def test_shared_grid_equals_manual_kron(self, random_key):
		kernel, W = self._kernel(random_key)
		x = jr.uniform(jr.fold_in(random_key, 1), (4, 1))

		K = kernel(x)
		B = W @ W.T + jnp.eye(3)  # default kappa = 1
		expected = jnp.kron(B, SEKernel(length_scale=1.0)(x, x))

		assert jnp.allclose(K, expected)

	@allure.title("ICMKernel: coregionalisation matrix is positive definite by construction")
	@allure.description(
		"W Wt alone is singular whenever n_latent < n_outputs; diag(kappa) with kappa > 0 "
		"makes B positive definite at any rank."
	)
	def test_coregionalisation_is_positive_definite(self, random_key):
		W = jr.normal(random_key, (5, 3))
		kernel = ICMKernel(SEKernel(length_scale=1.0), n_outputs=5, n_latent=3).replace(W=W)

		assert _min_eigval(W @ W.T) < 1e-6  # low rank: singular without kappa
		assert _min_eigval(kernel.coregionalisation) > 1e-6

	@allure.title("ICMKernel: shared-grid route matches heterotopic route on the same data")
	def test_shared_grid_matches_heterotopic_equivalent(self, random_key):
		kernel, W = self._kernel(random_key)
		x = jr.uniform(jr.fold_in(random_key, 1), (4, 1))
		N, P = x.shape[0], 3

		K_shared = kernel(x)
		output_ids = jnp.repeat(jnp.arange(P), N)
		x_tiled = jnp.tile(x, (P, 1))
		K_hetero = kernel(x_tiled, output_ids=output_ids, output_ids2=output_ids)

		assert jnp.allclose(K_shared, K_hetero)

	@allure.title("ICMKernel heterotopic: shuffled output_ids, independent per-pair check")
	def test_heterotopic_shuffled(self, random_key):
		W = jr.normal(random_key, (2, 2))
		base = SEKernel(length_scale=1.0)
		kernel = ICMKernel(base, n_outputs=2, n_latent=2).replace(W=W)
		B = W @ W.T + jnp.eye(2)  # default kappa = 1
		x = jnp.array([[0.1], [1.3], [0.4]])
		output_ids = jnp.array([1, 0, 1])

		K = kernel(x, output_ids=output_ids, output_ids2=output_ids)

		for i in range(3):
			for j in range(3):
				expected = B[output_ids[i], output_ids[j]] * base(x[i], x[j])
				assert jnp.allclose(K[i, j], expected)

	@allure.title("ICMKernel: validation errors")
	@allure.description(
		"n_outputs/n_latent must be positive; when output_ids is given and x2 too, "
		"output_ids2 is required (no symmetric fallback in that branch)."
	)
	def test_validation_errors(self):
		with pytest.raises(ValueError):
			ICMKernel(SEKernel(length_scale=1.0), n_outputs=0, n_latent=1)
		with pytest.raises(ValueError):
			ICMKernel(SEKernel(length_scale=1.0), n_outputs=2, n_latent=0)

		with pytest.raises(ValueError):
			ICMKernel(SEKernel(length_scale=1.0), n_outputs=2, n_latent=2, kappa=0.0)
		with pytest.raises(ValueError):
			ICMKernel(SEKernel(length_scale=1.0), n_outputs=2, n_latent=2, kappa=jnp.array([1.0, -1.0]))
		with pytest.raises(ValueError):
			ICMKernel(SEKernel(length_scale=1.0), n_outputs=2, n_latent=2, kappa=jnp.ones(3))

		kernel = ICMKernel(SEKernel(length_scale=1.0), n_outputs=2, n_latent=2)
		with pytest.raises(ValueError):
			kernel.replace(kappa=-1.0)

		x1 = jnp.array([[0.0], [1.0]])
		x2 = jnp.array([[0.5]])
		with pytest.raises(ValueError):
			kernel(x1, x2, output_ids=jnp.array([0, 1]))

	@allure.title("ICMKernel: replace(W=...) changes the coregionalisation matrix")
	def test_replace_w_changes_coregionalisation(self):
		kernel = ICMKernel(SEKernel(length_scale=1.0), n_outputs=2, n_latent=2)
		before = kernel.coregionalisation
		after = kernel.replace(W=jnp.array([[1.0, 0.0], [0.0, 2.0]])).coregionalisation
		assert not jnp.allclose(before, after)

	@allure.title("ICMKernel: kappa adds a per-output diagonal to B")
	@allure.description(
		"A scalar kappa is broadcast to (P,); a vector one gives each output its own value. "
		"kappa only ever touches B, so the whole within-output block of K is scaled by it, "
		"not just the diagonal of K."
	)
	@pytest.mark.parametrize("kappa", [2.0, jnp.array([0.5, 1.0, 4.0])])
	def test_kappa_adds_diagonal(self, random_key, kappa):
		W = jr.normal(random_key, (3, 2))
		base = SEKernel(length_scale=1.0)
		kernel = ICMKernel(base, n_outputs=3, n_latent=2, kappa=kappa).replace(W=W)

		assert kernel.kappa.shape == (3,)
		assert jnp.allclose(kernel.kappa, jnp.broadcast_to(jnp.asarray(kappa, dtype=float), (3,)))
		assert jnp.allclose(kernel.coregionalisation, W @ W.T + jnp.diag(kernel.kappa))

		x = jr.uniform(jr.fold_in(random_key, 1), (4, 1))
		assert jnp.allclose(kernel(x), jnp.kron(kernel.coregionalisation, base(x, x)))

	@allure.title("ICMKernel: replace(kappa=...) round-trips through the parametrisation")
	def test_replace_kappa_round_trip(self):
		kernel = ICMKernel(SEKernel(length_scale=1.0), n_outputs=3, n_latent=3)
		assert jnp.allclose(kernel.kappa, jnp.ones(3))

		kernel = kernel.replace(kappa=jnp.array([1.0, 2.0, 3.0]))
		assert jnp.allclose(kernel.kappa, jnp.array([1.0, 2.0, 3.0]))
		assert jnp.allclose(jnp.diag(kernel.coregionalisation - kernel.W @ kernel.W.T),
		                    jnp.array([1.0, 2.0, 3.0]))

	@allure.title("ICMKernel: a non-trainable kappa gets no gradient")
	@allure.description(
		"Holding kappa fixed during optimisation is what replaces an opt-out: the term stays "
		"in B, but stop_gradient keeps the optimiser from moving it."
	)
	def test_non_trainable_kappa_has_no_gradient(self):
		base = SEKernel(length_scale=1.0)
		x = jnp.linspace(0.0, 1.0, 4).reshape(-1, 1)

		def loss(kernel):
			return jnp.sum(kernel(x))

		trainable = ICMKernel(base, n_outputs=2, n_latent=2, kappa=2.0)
		frozen = ICMKernel(base, n_outputs=2, n_latent=2, kappa=2.0,
		                   kappa_parametrisation=NonTrainableParametrisation())

		assert not jnp.allclose(eqx.filter_grad(loss)(trainable)._kappa, 0.0)
		assert jnp.allclose(eqx.filter_grad(loss)(frozen)._kappa, 0.0)
		assert jnp.allclose(trainable.coregionalisation, frozen.coregionalisation)


class TestLCMKernel:
	"""Tests for LCMKernel (kernax/multioutput/LCMKernel.py)."""

	@allure.title("LCMKernel with one component equals a plain ICMKernel")
	def test_single_component_equals_icm(self, random_key):
		W = jr.normal(random_key, (3, 2))
		base = SEKernel(length_scale=1.0)
		lcm = LCMKernel([base], [W])
		icm = ICMKernel(base, n_outputs=3, n_latent=2).replace(W=W)

		x = jr.uniform(jr.fold_in(random_key, 1), (4, 1))
		assert jnp.allclose(lcm(x), icm(x))

	@allure.title("LCMKernel with two heterogeneous components equals their ICM sum")
	def test_two_components_equals_icm_sum(self, random_key):
		k1, k2 = SEKernel(length_scale=1.0), Matern32Kernel(length_scale=2.0)
		key1, key2 = jr.split(random_key)
		W1, W2 = jr.normal(key1, (3, 2)), jr.normal(key2, (3, 1))
		lcm = LCMKernel([k1, k2], [W1, W2])

		icm1 = ICMKernel(k1, n_outputs=3, n_latent=2).replace(W=W1)
		icm2 = ICMKernel(k2, n_outputs=3, n_latent=1).replace(W=W2)

		x = jr.uniform(jr.fold_in(random_key, 2), (4, 1))
		assert jnp.allclose(lcm(x), icm1(x) + icm2(x))

	@allure.title("LCMKernel: per-component kappas")
	@allure.description(
		"Each component carries its own kappa; omitting `kappas` gives every component the "
		"ICMKernel default of 1."
	)
	def test_per_component_kappas(self, random_key):
		k1, k2 = SEKernel(length_scale=1.0), Matern32Kernel(length_scale=2.0)
		W1, W2 = jnp.eye(2), jnp.ones((2, 1))
		lcm = LCMKernel([k1, k2], [W1, W2], kappas=[jnp.array([0.5, 1.5]), 3.0])

		assert jnp.allclose(lcm.components[0].kappa, jnp.array([0.5, 1.5]))
		assert jnp.allclose(lcm.components[1].kappa, jnp.full((2,), 3.0))

		icm1 = ICMKernel(k1, 2, 2, kappa=jnp.array([0.5, 1.5])).replace(W=W1)
		icm2 = ICMKernel(k2, 2, 1, kappa=3.0).replace(W=W2)
		x = jr.uniform(random_key, (4, 1))
		assert jnp.allclose(lcm(x), icm1(x) + icm2(x))

		default = LCMKernel([k1, k2], [W1, W2])
		assert all(jnp.allclose(c.kappa, jnp.ones(2)) for c in default.components)

	@allure.title("LCMKernel: validation errors")
	@allure.description(
		"kernels/matrices length mismatch, wrong W ndim, inconsistent n_outputs across "
		"components, empty kernel list."
	)
	def test_validation_errors(self):
		k = SEKernel(length_scale=1.0)
		with pytest.raises(ValueError):
			LCMKernel([], [])
		with pytest.raises(ValueError):
			LCMKernel([k], [jnp.eye(2), jnp.eye(2)])
		with pytest.raises(ValueError):
			LCMKernel([k], [jnp.ones(3)])
		with pytest.raises(ValueError):
			LCMKernel([k, k], [jnp.eye(2), jnp.eye(3)])
		with pytest.raises(ValueError):
			LCMKernel([k, k], [jnp.eye(2), jnp.eye(2)], kappas=[1.0])

	@allure.title("LCMKernel: replace() broadcasts to every component")
	def test_replace_broadcasts_to_components(self):
		k = SEKernel(length_scale=1.0)
		lcm = LCMKernel([k, k], [jnp.eye(2), jnp.eye(2)])
		lcm = lcm.replace(length_scale=2.0)
		assert lcm.components[0].inner.length_scale == 2.0
		assert lcm.components[1].inner.length_scale == 2.0


class TestConvolutionKernel:
	"""Tests for ConvolutionKernel (kernax/multioutput/ConvolutionKernel.py)."""

	@allure.title("ConvolutionKernel: self-covariance equals output variance")
	@allure.description("rho=1, distance=0 for a point against itself in the same output.")
	def test_self_covariance_equals_variance(self):
		variance = jnp.array([2.0, 3.0])
		bandwidth = jnp.array([1.0, 1.5])
		kernel = ConvolutionKernel(variance, bandwidth)

		x = jnp.array([[0.0]])
		K = kernel(x)  # shared-grid convenience path, shape (2, 2)

		assert jnp.allclose(jnp.diag(K), variance)

	@allure.title("ConvolutionKernel: cross-output correlation <= 1, equality iff same bandwidth")
	@allure.description("Docstring-stated property, checked directly rather than assumed.")
	def test_correlation_bound_and_equality_case(self, random_key):
		variance = jnp.ones(3)
		bandwidth = jr.uniform(random_key, (3,), minval=0.1, maxval=5.0)
		x = jnp.array([[0.0]])

		K = ConvolutionKernel(variance, bandwidth)(x)
		corr = K / jnp.sqrt(jnp.outer(variance, variance))
		assert jnp.all(corr <= 1.0 + 1e-6)

		K_eq = ConvolutionKernel(variance, jnp.full((3,), 2.0))(x)
		corr_eq = K_eq / jnp.sqrt(jnp.outer(variance, variance))
		assert jnp.allclose(corr_eq, 1.0)

	@allure.title("ConvolutionKernel: full assembled covariance is PSD")
	@pytest.mark.parametrize("ard", [False, True])
	def test_full_covariance_is_psd(self, random_key, ard):
		key1, key2 = jr.split(random_key)
		shape = (3, 2) if ard else (3,)
		variance = jr.uniform(key1, (3,), minval=0.5, maxval=2.0)
		bandwidth = jr.uniform(key2, shape, minval=0.5, maxval=2.0)
		kernel = ConvolutionKernel(variance, bandwidth, ard=ard)

		x = jr.uniform(jr.fold_in(random_key, 1), (4, 2 if ard else 1))
		K = kernel(x)

		assert _min_eigval(K) >= -1e-4

	@allure.title("ConvolutionKernel: shared-grid route matches manual tiling + output_ids")
	def test_shared_grid_matches_manual_tiling(self):
		variance = jnp.array([1.0, 2.0])
		bandwidth = jnp.array([1.0, 0.5])
		kernel = ConvolutionKernel(variance, bandwidth)
		x = jnp.array([[0.0], [1.0], [2.0]])
		N, P = x.shape[0], 2

		K_shared = kernel(x)
		output_ids = jnp.repeat(jnp.arange(P), N)
		x_tiled = jnp.tile(x, (P, 1))
		K_manual = kernel(x_tiled, output_ids=output_ids, output_ids2=output_ids)

		assert jnp.allclose(K_shared, K_manual)

	@allure.title("ConvolutionKernel: shape validation errors")
	def test_shape_validation_errors(self):
		with pytest.raises(ValueError):
			ConvolutionKernel(variance=jnp.ones(2), bandwidth=jnp.ones((2, 3)), ard=False)
		with pytest.raises(ValueError):
			ConvolutionKernel(variance=jnp.ones(3), bandwidth=jnp.ones(2), ard=False)


class TestMultiOutputCrossCutting:
	"""Invariants shared across the multi-output kernels."""

	@allure.title("Heterotopic cross-covariance is consistent under swapping x1/x2")
	@allure.description("K(x1, x2, ids1, ids2) must equal K(x2, x1, ids2, ids1).T for each kernel.")
	@pytest.mark.parametrize("kernel_name", ["block_diag", "icm", "convolution"])
	def test_swap_transpose_consistency(self, random_key, kernel_name):
		x1 = jnp.array([[0.1], [1.3], [0.4], [2.7]])
		x2 = jnp.array([[0.2], [1.1], [0.9]])
		ids1 = jnp.array([1, 0, 1, 0])
		ids2 = jnp.array([0, 1, 0])

		if kernel_name == "block_diag":
			base = SEKernel(length_scale=1.0)
			kernel = BlockDiagKernel(
				base, n_outputs=2, output_hps_in_axes=create_mask(base, default=None, length_scale=0)
			).replace(length_scale=jnp.array([0.5, 1.5]))
		elif kernel_name == "icm":
			W = jr.normal(random_key, (2, 2))
			kernel = ICMKernel(SEKernel(length_scale=1.0), n_outputs=2, n_latent=2).replace(W=W)
		else:
			kernel = ConvolutionKernel(variance=jnp.array([1.0, 2.0]), bandwidth=jnp.array([1.0, 0.5]))

		K12 = kernel(x1, x2, output_ids=ids1, output_ids2=ids2)
		K21 = kernel(x2, x1, output_ids=ids2, output_ids2=ids1)

		assert jnp.allclose(K12, K21.T)

	@allure.title("ICMKernel + BlockDiagKernel composition: correlated signal plus independent noise")
	@allure.description(
		"End-to-end smoke test that SumModule and the output_ids convention compose "
		"without shape errors, and that the resulting covariance stays PSD."
	)
	def test_icm_plus_block_diag_noise_composition(self, random_key):
		P = 3
		signal = ICMKernel(SEKernel(length_scale=1.0), n_outputs=P, n_latent=2)
		noise_base = WhiteNoiseKernel(noise=0.1)
		noise = BlockDiagKernel(
			noise_base, n_outputs=P, output_hps_in_axes=create_mask(noise_base, default=None, noise=0)
		).replace(noise=jnp.array([0.1, 0.2, 0.3]))
		gp_kernel = signal + noise

		x = jr.uniform(random_key, (5, 1))
		output_ids = jnp.array([0, 2, 1, 0, 2])

		K = gp_kernel(x, output_ids=output_ids, output_ids2=output_ids)

		assert K.shape == (5, 5)
		assert jnp.all(jnp.isfinite(K))
		assert jnp.allclose(K, K.T)
		assert _min_eigval(K) >= -1e-6
