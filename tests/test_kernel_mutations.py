"""
Tests for kernel mutation operations (replace method).
"""

import allure
import jax.numpy as jnp
import pytest
from equinox import EquinoxRuntimeError

from kernax import BatchKernel, ConstantKernel, PeriodicKernel, PolynomialKernel, SEKernel
from kernax.operators import SumKernel
from kernax.wrappers import ExpKernel


class TestReplaceMethod:
	"""Tests for the functional replace() API."""

	@allure.title("Replace constrained parameter")
	@allure.description("Test that replace() correctly transforms constrained parameters.")
	def test_replace_constrained(self):
		kernel = SEKernel(length_scale=1.0)
		new_kernel = kernel.replace(length_scale=2.0)

		assert kernel.length_scale == 1.0, "Original unchanged"
		assert new_kernel.length_scale == 2.0, "New value applied"
		assert kernel is not new_kernel, "Returns new instance"

	@allure.title("Replace unconstrained parameter")
	@allure.description("Test that replace() works with unconstrained parameters.")
	def test_replace_unconstrained(self):
		kernel = ConstantKernel(value=1.0)
		new_kernel = kernel.replace(value=-3.0)

		assert kernel.value == 1.0
		assert new_kernel.value == -3.0

	@allure.title("Replace multiple parameters")
	@allure.description("Test replacing multiple constrained parameters simultaneously.")
	def test_replace_multiple(self):
		kernel = PeriodicKernel(length_scale=1.0, variance=2.0, period=3.0)
		new_kernel = kernel.replace(length_scale=0.5, period=5.0)

		assert jnp.allclose(new_kernel.length_scale, 0.5)
		assert jnp.allclose(new_kernel.variance, 2.0)  # Unchanged
		assert jnp.allclose(new_kernel.period, 5.0)

	@allure.title("Replace validates constraints")
	@allure.description("Test that replace() rejects negative values for constrained parameters.")
	def test_replace_validation(self):
		kernel = SEKernel(length_scale=1.0)

		with pytest.raises(EquinoxRuntimeError):
			kernel.replace(length_scale=-1.0)

		with pytest.raises(EquinoxRuntimeError):
			kernel.replace(length_scale=0.0)


class TestReplaceWrapperKernel:
	"""Tests for replace() on WrapperKernels (ExpKernel, LogKernel, etc.)."""

	@allure.title("Replace inner kernel parameter in simple wrapper")
	@allure.description("Test that replace() forwards to inner_kernel for non-wrapper parameters.")
	def test_replace_inner_parameter(self):
		inner = SEKernel(length_scale=1.0)
		kernel = ExpKernel(inner)
		new_kernel = kernel.replace(length_scale=2.0)

		assert jnp.allclose(inner.length_scale, 1.0), "Original inner unchanged"
		assert jnp.allclose(new_kernel.inner_kernel.length_scale, 2.0)

	@allure.title("Replace inner kernel itself")
	@allure.description("Test that replace() can replace the inner_kernel attribute directly.")
	def test_replace_inner_kernel(self):
		kernel = ExpKernel(SEKernel(length_scale=1.0))
		new_inner = SEKernel(length_scale=3.0)
		new_kernel = kernel.replace(inner_kernel=new_inner)

		assert jnp.allclose(new_kernel.inner_kernel.length_scale, 3.0)

	@allure.title("Replace in nested wrappers")
	@allure.description("Test replace() with multiple levels of wrapping.")
	def test_replace_nested_wrappers(self):
		inner = SEKernel(length_scale=1.0)
		wrapped = ExpKernel(inner)
		double_wrapped = ExpKernel(wrapped)

		new_kernel = double_wrapped.replace(length_scale=2.0)

		# Should reach the innermost kernel
		assert jnp.allclose(new_kernel.inner_kernel.inner_kernel.length_scale, 2.0)


class TestReplaceBatchKernel:
	"""Tests for replace() on BatchKernel with broadcasting."""

	@allure.title("Replace with scalar broadcasts to batch dimension")
	@allure.description("Test that scalar values are automatically broadcast to batch size.")
	def test_replace_scalar_broadcasts(self):
		inner = SEKernel(length_scale=1.0)
		batch_kernel = BatchKernel(inner, batch_size=3, batch_in_axes=0)

		new_kernel = batch_kernel.replace(length_scale=2.0)

		# Should broadcast scalar to (3,)
		expected = jnp.array([2.0, 2.0, 2.0])
		assert new_kernel.inner_kernel._raw_length_scale.shape[0] == 3
		assert jnp.allclose(new_kernel.inner_kernel.length_scale, expected)

	@allure.title("Replace with correct batch dimensions")
	@allure.description("Test that values with correct batch dimensions are used directly.")
	def test_replace_correct_dimensions(self):
		inner = SEKernel(length_scale=1.0)
		batch_kernel = BatchKernel(inner, batch_size=3, batch_in_axes=0)

		new_values = jnp.array([1.0, 2.0, 3.0])
		new_kernel = batch_kernel.replace(length_scale=new_values)

		assert jnp.allclose(new_kernel.inner_kernel.length_scale, new_values)

	@allure.title("Replace with incorrect dimensions broadcasts")
	@allure.description("Test that values with wrong dimensions are broadcast to batch size.")
	def test_replace_broadcasts_wrong_dimensions(self):
		inner = SEKernel(length_scale=1.0)
		batch_kernel = BatchKernel(inner, batch_size=4, batch_in_axes=0)

		# Provide a single value that will be broadcast
		new_kernel = batch_kernel.replace(length_scale=2.5)

		expected = jnp.array([2.5, 2.5, 2.5, 2.5])
		assert jnp.allclose(new_kernel.inner_kernel.length_scale, expected)

	@allure.title("Replace shared parameters in BatchKernel")
	@allure.description("Test replace() when batch_in_axes=None (shared parameters).")
	def test_replace_shared_parameters(self):
		inner = SEKernel(length_scale=1.0)
		batch_kernel = BatchKernel(inner, batch_size=3, batch_in_axes=None)

		new_kernel = batch_kernel.replace(length_scale=2.0)

		# Shared parameter should remain scalar
		assert new_kernel.inner_kernel.length_scale.shape == ()
		assert jnp.allclose(new_kernel.inner_kernel.length_scale, 2.0)

	@allure.title("Replace with incompatible broadcast dimensions raises error")
	@allure.description("Test that incompatible array shapes raise a broadcasting error.")
	def test_replace_incompatible_broadcast_raises(self):
		inner = SEKernel(length_scale=1.0)
		batch_kernel = BatchKernel(inner, batch_size=3, batch_in_axes=0)

		# Try to replace with incompatible shape (2,) when batch_size is 3
		with pytest.raises((ValueError, RuntimeError)):
			batch_kernel.replace(length_scale=jnp.array([1.0, 2.0]))

		# Try with shape (4,) when batch_size is 3
		with pytest.raises((ValueError, RuntimeError)):
			batch_kernel.replace(length_scale=jnp.array([1.0, 2.0, 3.0, 4.0]))

	@allure.title("Replace in nested BatchKernels")
	@allure.description("Test replace() with SEKernel wrapped in two consecutive BatchKernels.")
	def test_replace_nested_batch_kernels(self):
		# Create nested BatchKernels: BatchKernel(BatchKernel(SEKernel))
		inner = SEKernel(length_scale=1.0)
		batch1 = BatchKernel(inner, batch_size=2, batch_in_axes=0)
		batch2 = BatchKernel(batch1, batch_size=3, batch_in_axes=0)

		# Replace with scalar - should broadcast to (3, 2)
		new_kernel = batch2.replace(length_scale=2.0)

		# Verify shape: outer batch (3,) wraps inner batch (2,)
		inner_kernel = new_kernel.inner_kernel.inner_kernel
		assert inner_kernel._raw_length_scale.shape == (3, 2)
		assert jnp.allclose(inner_kernel.length_scale, jnp.full((3, 2), 2.0))

		# Replace with array matching full nested shape (3, 2)
		new_values = jnp.array([[1.0, 1.5], [2.0, 2.5], [3.0, 3.5]])
		new_kernel = batch2.replace(length_scale=new_values)

		inner_kernel = new_kernel.inner_kernel.inner_kernel
		assert jnp.allclose(inner_kernel.length_scale, new_values)

		# Test that broadcasting from (1, 2) to (3, 2) works
		new_kernel = batch2.replace(length_scale=jnp.array([[1.0, 2.0]]))
		inner_kernel = new_kernel.inner_kernel.inner_kernel
		expected = jnp.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
		assert jnp.allclose(inner_kernel.length_scale, expected)


class TestReplaceOperatorKernel:
	"""Tests for replace() on OperatorKernels (SumKernel, ProductKernel)."""

	@allure.title("Replace common parameter in SE + SE")
	@allure.description("Test that common parameters are modified on both left and right kernels.")
	def test_replace_common_parameter_both_sides(self):
		left = SEKernel(length_scale=1.0)
		right = SEKernel(length_scale=2.0)
		kernel = SumKernel(left, right)

		new_kernel = kernel.replace(length_scale=3.0)

		# Both sides should be updated
		assert jnp.allclose(new_kernel.left_kernel.length_scale, 3.0)
		assert jnp.allclose(new_kernel.right_kernel.length_scale, 3.0)

	@allure.title("Replace parameter in SE + Polynomial")
	@allure.description("Test that parameter only on right kernel is modified, left kernel ignores it.")
	def test_replace_right_only_parameter(self):
		left = SEKernel(length_scale=1.0)
		right = PolynomialKernel(degree=2, gamma=1.0, constant=0.0)
		kernel = SumKernel(left, right)

		# Modify gamma (only exists on PolynomialKernel)
		new_kernel = kernel.replace(gamma=2.0)

		# Right kernel should be updated
		assert jnp.allclose(new_kernel.right_kernel.gamma, 2.0)
		# Left kernel should be unchanged (it doesn't have gamma)
		assert jnp.allclose(new_kernel.left_kernel.length_scale, 1.0)

	@allure.title("Replace parameter in SE + Exp(Polynomial)")
	@allure.description("Test that wrapping doesn't prevent parameter modification.")
	def test_replace_through_wrapper_in_composite(self):
		left = SEKernel(length_scale=1.0)
		right = ExpKernel(PolynomialKernel(degree=2, gamma=1.0, constant=0.0))
		kernel = SumKernel(left, right)

		# Modify gamma (exists in wrapped PolynomialKernel)
		new_kernel = kernel.replace(gamma=3.0)

		# Should reach through ExpKernel to PolynomialKernel
		assert jnp.allclose(new_kernel.right_kernel.inner_kernel.gamma, 3.0)
		# Left kernel unchanged
		assert jnp.allclose(new_kernel.left_kernel.length_scale, 1.0)

	@allure.title("Replace left_kernel in OperatorKernel")
	@allure.description("Test that left_kernel can be replaced directly.")
	def test_replace_left_kernel(self):
		kernel = SumKernel(SEKernel(length_scale=1.0), SEKernel(length_scale=2.0))
		new_left = SEKernel(length_scale=5.0)

		new_kernel = kernel.replace(left_kernel=new_left)

		assert jnp.allclose(new_kernel.left_kernel.length_scale, 5.0)
		assert jnp.allclose(new_kernel.right_kernel.length_scale, 2.0)

	@allure.title("Replace right_kernel in OperatorKernel")
	@allure.description("Test that right_kernel can be replaced directly.")
	def test_replace_right_kernel(self):
		kernel = SumKernel(SEKernel(length_scale=1.0), SEKernel(length_scale=2.0))
		new_right = PolynomialKernel(degree=3, gamma=2.0, constant=1.0)

		new_kernel = kernel.replace(right_kernel=new_right)

		assert jnp.allclose(new_kernel.left_kernel.length_scale, 1.0)
		assert new_kernel.right_kernel.degree == 3
		assert jnp.allclose(new_kernel.right_kernel.gamma, 2.0)