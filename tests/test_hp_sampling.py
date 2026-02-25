"""
Tests for the sample_hps_from_uniform_priors function.
"""

import allure
import jax
import jax.numpy as jnp
import pytest

from kernax import (
	AffineMean,
	BatchModule,
	Matern32Kernel,
	SEKernel,
	VarianceKernel,
	WhiteNoiseKernel,
	sample_hps_from_uniform_priors,
)
from kernax.other import ConstantKernel

KEY = jax.random.PRNGKey(0)


@allure.epic("HP Sampling")
class TestSampleHpsFromUniformPriors:
	"""Tests for sample_hps_from_uniform_priors."""

	# ── Bounds ──────────────────────────────────────────────────────────────

	@allure.title("Sampled constrained HP is within prior bounds")
	@allure.description(
		"A constrained HP (length_scale) must land in [low, high] after sampling."
	)
	def test_constrained_hp_within_bounds(self):
		kern = SEKernel(length_scale=1.0)
		sampled = sample_hps_from_uniform_priors(KEY, kern, {"length_scale": (2.0, 8.0)})
		assert 2.0 <= float(sampled.length_scale) <= 8.0

	@allure.title("Sampled unconstrained HP is within prior bounds")
	@allure.description(
		"An unconstrained HP (slope) must land in [low, high] after sampling."
	)
	def test_unconstrained_hp_within_bounds(self):
		mean = AffineMean(slope=0.0, intercept=0.0)
		sampled = sample_hps_from_uniform_priors(KEY, mean, {"slope": (-3.0, 3.0)})
		assert -3.0 <= float(sampled.slope) <= 3.0

	@allure.title("Multiple HPs all within their respective bounds")
	@allure.description(
		"When several HPs are specified in the priors dict, each must stay in its own bounds."
	)
	@pytest.mark.parametrize(
		"priors",
		[
			{"slope": (-1.0, 1.0), "intercept": (-5.0, 5.0)},
			{"slope": (0.0, 0.0), "intercept": (2.0, 2.0)},  # degenerate: bounds equal
		],
	)
	def test_multiple_hps_within_bounds(self, priors):
		mean = AffineMean(slope=0.0, intercept=0.0)
		sampled = sample_hps_from_uniform_priors(KEY, mean, priors)
		s_low, s_high = priors["slope"]
		i_low, i_high = priors["intercept"]
		assert s_low <= float(sampled.slope) <= s_high
		assert i_low <= float(sampled.intercept) <= i_high

	# ── Immutability ────────────────────────────────────────────────────────

	@allure.title("Original module is unchanged after sampling")
	@allure.description(
		"sample_hps_from_uniform_priors must not mutate the input module."
	)
	def test_original_unchanged(self):
		kern = SEKernel(length_scale=1.0)
		_ = sample_hps_from_uniform_priors(KEY, kern, {"length_scale": (2.0, 10.0)})
		assert jnp.allclose(kern.length_scale, jnp.array(1.0))

	@allure.title("Returns a new module instance")
	@allure.description("The returned module must not be the same object as the input.")
	def test_returns_new_instance(self):
		kern = SEKernel(length_scale=1.0)
		sampled = sample_hps_from_uniform_priors(KEY, kern, {"length_scale": (2.0, 10.0)})
		assert sampled is not kern

	# ── Unspecified HPs unchanged ────────────────────────────────────────────

	@allure.title("HP absent from priors dict is left unchanged")
	@allure.description(
		"HPs not listed in the priors dict must keep their original value."
	)
	def test_unspecified_hp_unchanged(self):
		kern = VarianceKernel(5.0) * SEKernel(length_scale=3.0)
		# Only variance is in priors; length_scale should be untouched
		sampled = sample_hps_from_uniform_priors(KEY, kern, {"variance": (1.0, 10.0)})
		assert jnp.allclose(
			kern.right._raw_length_scale,  # type: ignore[attr-defined]
			sampled.right._raw_length_scale,  # type: ignore[attr-defined]
		)

	@allure.title("Prior keys absent from the module are silently ignored")
	@allure.description(
		"A prior key that doesn't match any HP in the module must not raise an error."
	)
	def test_unknown_prior_key_ignored(self):
		kern = SEKernel(length_scale=1.0)
		# 'period' does not exist on SEKernel
		sampled = sample_hps_from_uniform_priors(KEY, kern, {"period": (1.0, 5.0)})
		# length_scale must be unchanged
		assert jnp.allclose(kern.length_scale, sampled.length_scale)

	# ── Nested / composite modules ──────────────────────────────────────────

	@allure.title("HPs are sampled across nested composite kernel")
	@allure.description(
		"In a ProductModule (VarianceKernel * SEKernel), both HPs are sampled."
	)
	def test_composite_kernel_all_hps_sampled(self):
		kern = VarianceKernel(5.0) * SEKernel(length_scale=3.0)
		priors = {"variance": (1.0, 10.0), "length_scale": (0.5, 5.0)}
		sampled = sample_hps_from_uniform_priors(KEY, kern, priors)
		assert 1.0 <= float(sampled.left.variance) <= 10.0  # type: ignore[attr-defined]
		assert 0.5 <= float(sampled.right.length_scale) <= 5.0  # type: ignore[attr-defined]

	@allure.title("HPs are sampled across a deeply nested kernel")
	@allure.description(
		"VarianceKernel * SEKernel + WhiteNoiseKernel: all three HPs must be in bounds."
	)
	def test_deep_composite_kernel(self):
		kern = VarianceKernel(2.0) * SEKernel(length_scale=4.0) + WhiteNoiseKernel(noise=0.1)
		priors = {"variance": (0.5, 5.0), "length_scale": (1.0, 8.0), "noise": (0.01, 1.0)}
		sampled = sample_hps_from_uniform_priors(KEY, kern, priors)

		# left = VarianceKernel * SEKernel, right = WhiteNoiseKernel
		assert 0.5 <= float(sampled.left.left.variance) <= 5.0  # type: ignore[attr-defined]
		assert 1.0 <= float(sampled.left.right.length_scale) <= 8.0  # type: ignore[attr-defined]
		assert 0.01 <= float(sampled.right.noise) <= 1.0  # type: ignore[attr-defined]

	@allure.title("HPs sampled independently in each sub-module")
	@allure.description(
		"When the same HP name appears in two sub-modules, both are sampled "
		"but with different values (independent draws)."
	)
	def test_independent_sampling_across_submodules(self):
		kern = SEKernel(length_scale=1.0) + Matern32Kernel(length_scale=1.0)
		sampled = sample_hps_from_uniform_priors(KEY, kern, {"length_scale": (1.0, 10.0)})
		ls_left = float(sampled.left.length_scale)  # type: ignore[attr-defined]
		ls_right = float(sampled.right.length_scale)  # type: ignore[attr-defined]
		assert ls_left != ls_right, "Different sub-modules should have different sampled values"

	# ── Batched HPs ──────────────────────────────────────────────────────────

	@allure.title("Batched HP array shape is preserved")
	@allure.description(
		"When an HP is batched (shape (B,)), the returned HP must have the same shape."
	)
	def test_batched_hp_shape_preserved(self):
		base = SEKernel(length_scale=1.0)
		batched = BatchModule(base, batch_size=4, batch_in_axes=0)
		sampled = sample_hps_from_uniform_priors(KEY, batched, {"length_scale": (1.0, 5.0)})
		assert sampled.inner.length_scale.shape == (4,)  # type: ignore[attr-defined]

	@allure.title("Each element of a batched HP is sampled within bounds")
	@allure.description(
		"All B elements of a batched HP array must respect [low, high]."
	)
	def test_batched_hp_all_elements_within_bounds(self):
		base = SEKernel(length_scale=1.0)
		batched = BatchModule(base, batch_size=8, batch_in_axes=0)
		sampled = sample_hps_from_uniform_priors(KEY, batched, {"length_scale": (2.0, 6.0)})
		ls = sampled.inner.length_scale  # type: ignore[attr-defined]
		assert jnp.all(ls >= 2.0) and jnp.all(ls <= 6.0)

	@allure.title("Batched HP elements are independently sampled")
	@allure.description(
		"The B elements of a batched HP must not all be identical."
	)
	def test_batched_hp_elements_differ(self):
		base = SEKernel(length_scale=1.0)
		batched = BatchModule(base, batch_size=8, batch_in_axes=0)
		sampled = sample_hps_from_uniform_priors(KEY, batched, {"length_scale": (1.0, 10.0)})
		ls = sampled.inner.length_scale  # type: ignore[attr-defined]
		assert not jnp.all(ls == ls[0]), "Batch elements should differ"

	# ── Determinism ──────────────────────────────────────────────────────────

	@allure.title("Same key yields same sampled values")
	@allure.description(
		"Two calls with the same PRNG key must produce identical results."
	)
	def test_deterministic_with_same_key(self):
		kern = SEKernel(length_scale=1.0)
		s1 = sample_hps_from_uniform_priors(KEY, kern, {"length_scale": (1.0, 5.0)})
		s2 = sample_hps_from_uniform_priors(KEY, kern, {"length_scale": (1.0, 5.0)})
		assert jnp.allclose(s1.length_scale, s2.length_scale)

	@allure.title("Different keys yield different sampled values")
	@allure.description(
		"Two calls with different PRNG keys must (almost surely) produce different results."
	)
	def test_different_keys_give_different_values(self):
		kern = SEKernel(length_scale=1.0)
		key1, key2 = jax.random.split(KEY)
		s1 = sample_hps_from_uniform_priors(key1, kern, {"length_scale": (1.0, 10.0)})
		s2 = sample_hps_from_uniform_priors(key2, kern, {"length_scale": (1.0, 10.0)})
		assert not jnp.allclose(s1.length_scale, s2.length_scale)

	# ── Edge cases ───────────────────────────────────────────────────────────

	@allure.title("Empty priors dict leaves module unchanged")
	@allure.description("Calling with an empty priors dict must return an equivalent module.")
	def test_empty_priors(self):
		kern = SEKernel(length_scale=3.0)
		sampled = sample_hps_from_uniform_priors(KEY, kern, {})
		assert jnp.allclose(kern.length_scale, sampled.length_scale)

	@allure.title("Function is exported from the top-level kernax namespace")
	@allure.description(
		"sample_hps_from_uniform_priors must be accessible via `import kernax`."
	)
	def test_exported_from_kernax(self):
		import kernax
		assert hasattr(kernax, "sample_hps_from_uniform_priors")
		assert callable(kernax.sample_hps_from_uniform_priors)