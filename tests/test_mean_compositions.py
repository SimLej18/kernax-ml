"""
Tests for mean function compositions (SumModule, ProductModule, NegModule on means).
"""

import allure
import jax.numpy as jnp
import pytest

from kernax import (
	AffineMean,
	ConstantMean,
	LinearMean,
	NegModule,
	ProductModule,
	SumModule,
	ZeroMean,
)


@pytest.fixture
def sample_input():
	"""Batch of 1D input points (N, 1)."""
	return jnp.array([[0.0], [1.0], [2.0], [3.0], [4.0]])


class TestMeanAddition:
	"""Tests for mean addition via SumModule."""

	@allure.title("SumModule instantiation with two means")
	def test_sum_module_with_means(self, sample_input):
		m1 = ZeroMean()
		m2 = ConstantMean(constant=2.0)
		combined = m1 + m2
		assert isinstance(combined, SumModule)
		result = combined(sample_input)
		expected = m1(sample_input) + m2(sample_input)
		assert jnp.allclose(result, expected)

	@allure.title("Mean radd with scalar auto-converts to ConstantMean")
	def test_radd_scalar(self, sample_input):
		m = LinearMean(slope=1.0)
		combined = 3.0 + m
		result = combined(sample_input)
		expected = 3.0 + m(sample_input)
		assert jnp.allclose(result, expected)

	@allure.title("LinearMean + ConstantMean equals AffineMean")
	def test_linear_plus_constant_equals_affine(self, sample_input):
		slope, intercept = 2.0, 3.0
		composed = LinearMean(slope=slope) + ConstantMean(constant=intercept)
		reference = AffineMean(slope=slope, intercept=intercept)
		assert jnp.allclose(composed(sample_input), reference(sample_input))


class TestMeanSubtraction:
	"""Tests for mean subtraction."""

	@allure.title("Mean subtraction")
	def test_mean_sub(self, sample_input):
		m1 = ConstantMean(constant=5.0)
		m2 = ConstantMean(constant=2.0)
		combined = m1 - m2
		result = combined(sample_input)
		expected = m1(sample_input) - m2(sample_input)
		assert jnp.allclose(result, expected)


class TestMeanMultiplication:
	"""Tests for mean multiplication via ProductModule."""

	@allure.title("ProductModule with two means")
	def test_mean_mul(self, sample_input):
		m1 = ConstantMean(constant=2.0)
		m2 = LinearMean(slope=3.0)
		combined = m1 * m2
		assert isinstance(combined, ProductModule)
		result = combined(sample_input)
		expected = m1(sample_input) * m2(sample_input)
		assert jnp.allclose(result, expected)

	@allure.title("Mean rmul with scalar auto-converts to ConstantMean")
	def test_rmul_scalar(self, sample_input):
		m = LinearMean(slope=1.0)
		combined = 2.0 * m
		result = combined(sample_input)
		expected = 2.0 * m(sample_input)
		assert jnp.allclose(result, expected)


class TestMeanNegation:
	"""Tests for mean negation via NegModule."""

	@allure.title("NegModule on a mean")
	def test_neg(self, sample_input):
		m = ConstantMean(constant=3.0)
		negated = -m
		assert isinstance(negated, NegModule)
		result = negated(sample_input)
		assert jnp.allclose(result, -m(sample_input))


class TestComplexMeanCompositions:
	"""Tests for complex mean compositions."""

	@allure.title("2 * LinearMean + ConstantMean equals AffineMean")
	def test_complex_chain(self, sample_input):
		composed = 2.0 * LinearMean(slope=1.0) + ConstantMean(constant=1.0)
		reference = AffineMean(slope=2.0, intercept=1.0)
		assert jnp.allclose(composed(sample_input), reference(sample_input))

	@allure.title("String representation of composed means is valid")
	def test_str_representation(self):
		m1 = LinearMean(slope=1.0)
		m2 = ConstantMean(constant=2.0)

		assert isinstance(str(m1 + m2), str)
		assert len(str(m1 + m2)) > 0
		assert isinstance(str(m1 * m2), str)
		assert isinstance(str(-m1), str)
		assert isinstance(str(m1 - m2), str)