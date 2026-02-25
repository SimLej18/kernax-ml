"""
Tests for mean function mutation operations (replace method).
"""

import allure
import jax.numpy as jnp

import pytest

from kernax import (
	AffineMean,
	BatchModule,
	ConstantMean,
	ExpModule,
	LinearMean,
	SumModule,
)


class TestReplaceMethod:
	"""Tests for the functional replace() API on base mean functions."""

	@allure.title("Replace constant parameter on ConstantMean")
	@allure.description("Test that replace() correctly modifies the 'constant' parameter.")
	def test_replace_constant(self):
		mean = ConstantMean(constant=3.0)
		new_mean = mean.replace(constant=5.0)

		assert jnp.allclose(mean.constant, 3.0), "Original unchanged"
		assert jnp.allclose(new_mean.constant, 5.0), "New value applied"
		assert mean is not new_mean, "Returns new instance"

	@allure.title("Replace slope on LinearMean")
	@allure.description("Test that replace() correctly modifies the 'slope' parameter.")
	def test_replace_slope(self):
		mean = LinearMean(slope=2.0)
		new_mean = mean.replace(slope=4.0)

		assert jnp.allclose(mean.slope, 2.0), "Original unchanged"
		assert jnp.allclose(new_mean.slope, 4.0), "New value applied"

	@allure.title("Mean parameters accept negative values")
	@allure.description("Test that mean parameters have no positivity constraint.")
	def test_replace_allows_negative(self):
		mean = LinearMean(slope=1.0)
		new_mean = mean.replace(slope=-2.0)
		assert jnp.allclose(new_mean.slope, -2.0)

	@allure.title("Replace multiple parameters simultaneously on AffineMean")
	@allure.description("Test replacing slope and intercept at once, and partial replacement.")
	def test_replace_multiple(self):
		mean = AffineMean(slope=1.0, intercept=0.0)
		new_mean = mean.replace(slope=3.0, intercept=2.0)

		assert jnp.allclose(new_mean.slope, 3.0)
		assert jnp.allclose(new_mean.intercept, 2.0)

		# Partial replacement leaves other params unchanged
		partial = mean.replace(slope=5.0)
		assert jnp.allclose(partial.slope, 5.0)
		assert jnp.allclose(partial.intercept, 0.0)


class TestReplaceWrapperMean:
	"""Tests for replace() on WrapperModules wrapping means."""

	@allure.title("Replace inner mean parameter through ExpModule")
	@allure.description("Test that replace() forwards to inner mean for non-wrapper parameters.")
	def test_replace_inner_parameter(self):
		inner = LinearMean(slope=1.0)
		wrapped = ExpModule(inner)
		new_wrapped = wrapped.replace(slope=2.0)

		assert jnp.allclose(inner.slope, 1.0), "Original inner unchanged"
		assert jnp.allclose(new_wrapped.inner.slope, 2.0)

	@allure.title("Replace inner mean directly")
	@allure.description("Test that replace() can swap out the inner attribute.")
	def test_replace_inner(self):
		wrapped = ExpModule(LinearMean(slope=1.0))
		new_inner = LinearMean(slope=5.0)
		new_wrapped = wrapped.replace(inner=new_inner)

		assert jnp.allclose(new_wrapped.inner.slope, 5.0)

	@allure.title("Replace in nested wrappers")
	@allure.description("Test replace() propagates through multiple wrapping levels.")
	def test_replace_nested_wrappers(self):
		double_wrapped = ExpModule(ExpModule(ConstantMean(constant=1.0)))
		new_wrapped = double_wrapped.replace(constant=3.0)

		assert jnp.allclose(new_wrapped.inner.inner.constant, 3.0)


class TestReplaceOperatorMean:
	"""Tests for replace() on OperatorModules (SumModule, ProductModule) containing means."""

	@allure.title("Replace common parameter in LinearMean + LinearMean")
	@allure.description("Test that common parameters are updated on both left and right sides.")
	def test_replace_common_parameter_both_sides(self):
		left = LinearMean(slope=1.0)
		right = LinearMean(slope=2.0)
		composed = SumModule(left, right)

		new_composed = composed.replace(slope=5.0)

		assert jnp.allclose(new_composed.left.slope, 5.0)
		assert jnp.allclose(new_composed.right.slope, 5.0)

	@allure.title("Replace parameter only present on right side")
	@allure.description("Test that a parameter absent from left is only updated on the right.")
	def test_replace_right_only_parameter(self):
		left = LinearMean(slope=1.0)
		right = AffineMean(slope=2.0, intercept=3.0)
		composed = SumModule(left, right)

		new_composed = composed.replace(intercept=10.0)

		assert jnp.allclose(new_composed.right.intercept, 10.0)
		assert jnp.allclose(new_composed.left.slope, 1.0)  # Left unchanged

	@allure.title("Replace left mean in OperatorModule")
	@allure.description("Test that the left sub-mean can be replaced directly.")
	def test_replace_left(self):
		composed = SumModule(LinearMean(slope=1.0), ConstantMean(constant=2.0))
		new_composed = composed.replace(left=LinearMean(slope=5.0))

		assert jnp.allclose(new_composed.left.slope, 5.0)
		assert jnp.allclose(new_composed.right.constant, 2.0)

	@allure.title("Replace right mean in OperatorModule")
	@allure.description("Test that the right sub-mean can be replaced directly.")
	def test_replace_right(self):
		composed = SumModule(LinearMean(slope=1.0), ConstantMean(constant=2.0))
		new_composed = composed.replace(right=AffineMean(slope=3.0, intercept=1.0))

		assert jnp.allclose(new_composed.left.slope, 1.0)
		assert jnp.allclose(new_composed.right.slope, 3.0)
		assert jnp.allclose(new_composed.right.intercept, 1.0)

	@allure.title("Replace through wrapper in composite mean")
	@allure.description("Test that wrapping doesn't prevent parameter modification in a sum.")
	def test_replace_through_wrapper_in_composite(self):
		left = LinearMean(slope=1.0)
		right = ExpModule(AffineMean(slope=2.0, intercept=0.0))
		composed = SumModule(left, right)

		new_composed = composed.replace(intercept=3.0)

		# Should reach through ExpModule to AffineMean
		assert jnp.allclose(new_composed.right.inner.intercept, 3.0)
		assert jnp.allclose(new_composed.left.slope, 1.0)


class TestReplaceBatchMean:
	"""Tests for replace() on BatchModule wrapping a mean."""

	@allure.title("Replace with scalar broadcasts to batch dimension")
	@allure.description("Test that a scalar value is automatically broadcast to batch size.")
	def test_replace_scalar_broadcasts(self):
		batch_mean = BatchModule(ConstantMean(constant=1.0), batch_size=3, batch_in_axes=0)

		new_mean = batch_mean.replace(constant=5.0)

		assert new_mean.inner.constant.shape[0] == 3
		assert jnp.allclose(new_mean.inner.constant, jnp.array([5.0, 5.0, 5.0]))

	@allure.title("Replace with correct batch dimensions")
	@allure.description("Test that a vector with matching batch size is applied directly.")
	def test_replace_correct_dimensions(self):
		batch_mean = BatchModule(ConstantMean(constant=1.0), batch_size=3, batch_in_axes=0)

		new_values = jnp.array([1.0, 2.0, 3.0])
		new_mean = batch_mean.replace(constant=new_values)

		assert jnp.allclose(new_mean.inner.constant, new_values)

	@allure.title("Replace shared parameters in BatchModule")
	@allure.description("Test replace() when batch_in_axes=None (shared parameters).")
	def test_replace_shared_parameters(self):
		batch_mean = BatchModule(LinearMean(slope=1.0), batch_size=3, batch_in_axes=None)

		new_mean = batch_mean.replace(slope=4.0)

		assert new_mean.inner.slope.shape == ()
		assert jnp.allclose(new_mean.inner.slope, 4.0)

	@allure.title("BatchModule batch_size is immutable")
	@allure.description("Test that attempting to modify batch_size raises a ValueError.")
	def test_batch_size_immutable(self):
		batch_mean = BatchModule(ConstantMean(constant=1.0), batch_size=3, batch_in_axes=0)

		with pytest.raises(ValueError, match="batch_size"):
			batch_mean.replace(batch_size=5)

	@allure.title("BatchModule batch_in_axes is immutable")
	@allure.description("Test that attempting to modify batch_in_axes raises a ValueError.")
	def test_batch_in_axes_immutable(self):
		batch_mean = BatchModule(LinearMean(slope=1.0), batch_size=3, batch_in_axes=0)

		with pytest.raises(ValueError, match="batch_in_axes"):
			batch_mean.replace(batch_in_axes=None)