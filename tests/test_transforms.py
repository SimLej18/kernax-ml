"""
Tests for the kernax parameter transformation utilities.
"""

import warnings

import allure
import jax.numpy as jnp
import pytest

import kernax


@allure.title("Test identity transform is bijective")
@allure.description("Verify that identity transform is its own inverse")
def test_identity_transform():
	"""Test that identity transform works correctly."""
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", RuntimeWarning)
		kernax.config.unsafe_reset()

	kernax.config.parameter_transform = "identity"

	# Test with single value
	value = jnp.array(2.0)
	unconstrained = kernax.transforms.to_unconstrained(value)
	constrained = kernax.transforms.to_constrained(unconstrained)

	assert jnp.allclose(unconstrained, value)
	assert jnp.allclose(constrained, value)

	# Test with array
	values = jnp.array([0.5, 1.0, 2.0, 5.0])
	unconstrained = kernax.transforms.to_unconstrained(values)
	constrained = kernax.transforms.to_constrained(unconstrained)

	assert jnp.allclose(unconstrained, values)
	assert jnp.allclose(constrained, values)

	with warnings.catch_warnings():
		warnings.simplefilter("ignore", RuntimeWarning)
		kernax.config.unsafe_reset()


@allure.title("Test exp transform is bijective")
@allure.description("Verify that exp transform correctly inverts")
def test_exp_transform():
	"""Test that exp transform works correctly."""
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", RuntimeWarning)
		kernax.config.unsafe_reset()

	kernax.config.parameter_transform = "exp"

	# Test with single value
	value = jnp.array(2.0)
	unconstrained = kernax.transforms.to_unconstrained(value)
	constrained = kernax.transforms.to_constrained(unconstrained)

	# Unconstrained should be log(value)
	assert jnp.allclose(unconstrained, jnp.log(value))
	# Constrained should recover original
	assert jnp.allclose(constrained, value)

	# Test with array
	values = jnp.array([0.5, 1.0, 2.0, 5.0])
	unconstrained = kernax.transforms.to_unconstrained(values)
	constrained = kernax.transforms.to_constrained(unconstrained)

	assert jnp.allclose(unconstrained, jnp.log(values))
	assert jnp.allclose(constrained, values)

	with warnings.catch_warnings():
		warnings.simplefilter("ignore", RuntimeWarning)
		kernax.config.unsafe_reset()


@allure.title("Test softplus transform is bijective")
@allure.description("Verify that softplus transform correctly inverts")
def test_softplus_transform():
	"""Test that softplus transform works correctly."""
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", RuntimeWarning)
		kernax.config.unsafe_reset()

	kernax.config.parameter_transform = "softplus"

	# Test with single value
	value = jnp.array(2.0)
	unconstrained = kernax.transforms.to_unconstrained(value)
	constrained = kernax.transforms.to_constrained(unconstrained)

	# Constrained should recover original
	assert jnp.allclose(constrained, value, rtol=1e-5)

	# Test with array
	values = jnp.array([0.5, 1.0, 2.0, 5.0])
	unconstrained = kernax.transforms.to_unconstrained(values)
	constrained = kernax.transforms.to_constrained(unconstrained)

	assert jnp.allclose(constrained, values, rtol=1e-5)

	with warnings.catch_warnings():
		warnings.simplefilter("ignore", RuntimeWarning)
		kernax.config.unsafe_reset()


@allure.title("Test to_constrained always returns positive values")
@allure.description("Verify that all transforms produce positive outputs")
@pytest.mark.parametrize("transform", ["identity", "exp", "softplus"])
def test_to_constrained_always_positive(transform):
	"""Test that to_constrained always returns positive values."""
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", RuntimeWarning)
		kernax.config.unsafe_reset()

	kernax.config.parameter_transform = transform

	# Test with negative unconstrained values (except identity)
	if transform == "identity":
		# Identity requires positive input
		unconstrained_values = jnp.array([0.1, 1.0, 5.0])
	else:
		# Exp and softplus can handle negative unconstrained values
		unconstrained_values = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])

	constrained = kernax.transforms.to_constrained(unconstrained_values)

	# All constrained values should be positive
	assert jnp.all(constrained > 0)

	with warnings.catch_warnings():
		warnings.simplefilter("ignore", RuntimeWarning)
		kernax.config.unsafe_reset()


@allure.title("Test transforms with SEKernel")
@allure.description("Verify that SEKernel correctly uses transforms")
@pytest.mark.parametrize("transform", ["identity", "exp", "softplus"])
def test_sekernel_with_transforms(transform):
	"""Test that SEKernel works with all transforms."""
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", RuntimeWarning)
		kernax.config.unsafe_reset()

	kernax.config.parameter_transform = transform

	# Create kernel
	length_scale_input = 2.0
	kernel = kernax.SEKernel(length_scale=length_scale_input)

	# Retrieved length_scale should match input
	assert jnp.allclose(kernel.length_scale, length_scale_input, rtol=1e-5)

	# Kernel computation should work
	x1 = jnp.array([0.0])
	x2 = jnp.array([1.0])
	cov = kernel(x1, x2)

	# Should be a valid covariance value
	assert jnp.isfinite(cov)
	assert cov > 0  # SE kernel is always positive

	with warnings.catch_warnings():
		warnings.simplefilter("ignore", RuntimeWarning)
		kernax.config.unsafe_reset()


@allure.title("Test transforms preserve kernel behavior")
@allure.description("Verify that transforms don't change kernel computation")
def test_transform_preserves_kernel_behavior():
	"""
	Test that a kernel created with a transform produces correct covariances.

	Note: We cannot test multiple transforms in one session because config.parameter_transform
	is a global setting that affects how kernels interpret their stored parameters. This is
	by design - the transform must be set before creating kernels and remain fixed.
	"""
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", RuntimeWarning)
		kernax.config.unsafe_reset()

	# Use identity transform for simplicity
	kernax.config.parameter_transform = "identity"

	length_scale = 2.0
	kernel = kernax.SEKernel(length_scale=length_scale)

	# Verify length_scale is correct
	assert jnp.allclose(kernel.length_scale, length_scale)

	# Test kernel computation
	x1 = jnp.array([0.0])
	x2 = jnp.array([1.0])
	cov = kernel(x1, x2)

	# Expected: exp(-0.5 * (0-1)^2 / 2^2) = exp(-0.5 * 1 / 4) = exp(-0.125)
	expected = jnp.exp(-0.125)
	assert jnp.allclose(cov, expected, rtol=1e-5)

	# Clean up
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", RuntimeWarning)
		kernax.config.unsafe_reset()
