"""
Tests for base mean function implementations (ZeroMean, ConstantMean, LinearMean, AffineMean).
"""

import allure
import jax.numpy as jnp
import pytest

from kernax import (
	AffineMean,
	ConstantMean,
	LinearMean,
	ZeroMean,
)


@pytest.fixture
def sample_1d_input():
	"""1D input vector (N, 1)."""
	return jnp.array([[0.0], [1.0], [2.0], [3.0]])


@pytest.fixture
def sample_scalar_input():
	"""Single 1D input point."""
	return jnp.array([2.0])


class TestZeroMean:
	"""Tests for ZeroMean."""

	@allure.title("ZeroMean instantiation and string representation")
	def test_instantiation(self):
		mean = ZeroMean()
		assert mean is not None
		assert isinstance(str(mean), str)

	@allure.title("ZeroMean scalar computation")
	def test_scalar_computation(self, sample_scalar_input):
		mean = ZeroMean()
		result = mean(sample_scalar_input)
		assert jnp.allclose(result, jnp.array(0.0))

	@allure.title("ZeroMean vector computation")
	def test_vector_computation(self, sample_1d_input):
		mean = ZeroMean()
		result = mean(sample_1d_input)
		assert result.shape == (sample_1d_input.shape[0],)
		assert jnp.allclose(result, jnp.zeros(sample_1d_input.shape[0]))


class TestConstantMean:
	"""Tests for ConstantMean."""

	@allure.title("ConstantMean instantiation and string representation")
	def test_instantiation(self):
		mean = ConstantMean(constant=3.0)
		assert mean is not None
		assert isinstance(str(mean), str)

	@pytest.mark.parametrize("constant", [0.0, 1.0, -2.5, 100.0])
	@allure.title("ConstantMean scalar computation")
	def test_scalar_computation(self, sample_scalar_input, constant):
		mean = ConstantMean(constant=constant)
		result = mean(sample_scalar_input)
		assert jnp.allclose(result, jnp.array(constant))

	@allure.title("ConstantMean vector computation")
	def test_vector_computation(self, sample_1d_input):
		mean = ConstantMean(constant=5.0)
		result = mean(sample_1d_input)
		assert result.shape == (sample_1d_input.shape[0],)
		assert jnp.allclose(result, jnp.full(sample_1d_input.shape[0], 5.0))


class TestLinearMean:
	"""Tests for LinearMean."""

	@allure.title("LinearMean instantiation and string representation")
	def test_instantiation(self):
		mean = LinearMean(slope=2.0)
		assert mean is not None
		assert isinstance(str(mean), str)

	@pytest.mark.parametrize("slope,x", [(1.0, 3.0), (2.0, 1.5), (-1.0, 4.0), (0.5, 2.0)])
	@allure.title("LinearMean scalar computation")
	def test_scalar_computation(self, slope, x):
		mean = LinearMean(slope=slope)
		result = mean(jnp.array([x]))
		assert jnp.allclose(result, jnp.array(slope * x))

	@allure.title("LinearMean vector computation")
	def test_vector_computation(self, sample_1d_input):
		slope = 3.0
		mean = LinearMean(slope=slope)
		result = mean(sample_1d_input)
		expected = (slope * sample_1d_input).squeeze(-1)
		assert result.shape == (sample_1d_input.shape[0],)
		assert jnp.allclose(result, expected)

	@allure.title("LinearMean 2D input with scalar slope")
	def test_2d_scalar_slope(self):
		x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
		mean = LinearMean(slope=2.0)
		result = mean(x)
		expected = jnp.array([2.0 * 1 + 2.0 * 2, 2.0 * 3 + 2.0 * 4])  # [6, 14]
		assert result.shape == (x.shape[0],)
		assert jnp.allclose(result, expected)

	@allure.title("LinearMean 2D input with vector slope")
	def test_2d_vector_slope(self):
		x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
		mean = LinearMean(slope=jnp.array([1.0, 2.0]))
		result = mean(x)
		expected = jnp.array([1.0 * 1 + 2.0 * 2, 1.0 * 3 + 2.0 * 4])  # [5, 11]
		assert result.shape == (x.shape[0],)
		assert jnp.allclose(result, expected)


class TestAffineMean:
	"""Tests for AffineMean."""

	@allure.title("AffineMean instantiation and string representation")
	def test_instantiation(self):
		mean = AffineMean(slope=2.0, intercept=1.0)
		assert mean is not None
		assert isinstance(str(mean), str)

	@pytest.mark.parametrize("slope,intercept,x", [
		(1.0, 0.0, 3.0),
		(2.0, 1.0, 1.5),
		(-1.0, 5.0, 4.0),
		(0.0, 3.0, 2.0),
	])
	@allure.title("AffineMean scalar computation")
	def test_scalar_computation(self, slope, intercept, x):
		mean = AffineMean(slope=slope, intercept=intercept)
		result = mean(jnp.array([x]))
		assert jnp.allclose(result, jnp.array(slope * x + intercept))

	@allure.title("AffineMean vector computation")
	def test_vector_computation(self, sample_1d_input):
		slope, intercept = 2.0, -1.0
		mean = AffineMean(slope=slope, intercept=intercept)
		result = mean(sample_1d_input)
		expected = (slope * sample_1d_input + intercept).squeeze(-1)
		assert result.shape == (sample_1d_input.shape[0],)
		assert jnp.allclose(result, expected)

	@allure.title("AffineMean with zero slope equals ConstantMean")
	def test_zero_slope_equals_constant_mean(self, sample_1d_input):
		intercept = 4.2
		affine = AffineMean(slope=0.0, intercept=intercept)
		constant = ConstantMean(constant=intercept)
		assert jnp.allclose(affine(sample_1d_input), constant(sample_1d_input))

	@allure.title("AffineMean 2D input with scalar slope and intercept")
	def test_2d_scalar_slope_intercept(self):
		x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
		mean = AffineMean(slope=2.0, intercept=1.0)
		result = mean(x)
		expected = jnp.array([2.0 * 1 + 2.0 * 2 + 1.0, 2.0 * 3 + 2.0 * 4 + 1.0])  # [7, 15]
		assert result.shape == (x.shape[0],)
		assert jnp.allclose(result, expected)

	@allure.title("AffineMean 2D input with vector slope and scalar intercept")
	def test_2d_vector_slope_scalar_intercept(self):
		x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
		mean = AffineMean(slope=jnp.array([1.0, 2.0]), intercept=0.5)
		result = mean(x)
		expected = jnp.array([1.0 * 1 + 2.0 * 2 + 0.5, 1.0 * 3 + 2.0 * 4 + 0.5])  # [5.5, 11.5]
		assert result.shape == (x.shape[0],)
		assert jnp.allclose(result, expected)