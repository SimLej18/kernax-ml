"""
Tests for wrapper modules applied to mean functions (ExpModule, LogModule, ActiveDimsModule, BatchModule).

Note: BlockKernel, BlockDiagKernel, and ARDKernel are kernel-specific and are not tested here.
"""

import allure
import jax.numpy as jnp

from kernax import (
	ActiveDimsModule,
	BatchModule,
	ConstantMean,
	ExpModule,
	LinearMean,
	LogModule,
	ZeroMean,
)


class TestExpModuleWithMean:
	"""Tests for ExpModule wrapping a mean function."""

	@allure.title("ExpModule wrapping a mean: instantiation")
	@allure.description("Test that ExpModule can wrap a mean function.")
	def test_instantiation(self):
		mean = ConstantMean(constant=1.0)
		wrapped = ExpModule(mean)
		assert wrapped.inner is mean

	@allure.title("ExpModule wrapping ConstantMean: scalar computation")
	@allure.description("Test exp(constant) for a scalar input.")
	def test_scalar_computation(self):
		c = 2.0
		wrapped = ExpModule(ConstantMean(constant=c))
		result = wrapped(jnp.array([0.5]))
		assert jnp.allclose(result, jnp.exp(jnp.array(c)))

	@allure.title("ExpModule wrapping LinearMean: vector computation")
	@allure.description("Test exp(slope * x) for a batch of inputs.")
	def test_vector_computation(self):
		slope = 1.0
		wrapped = ExpModule(LinearMean(slope=slope))
		x = jnp.array([[0.0], [1.0], [2.0]])
		result = wrapped(x)
		expected = jnp.exp(slope * x.squeeze(-1))
		assert result.shape == (x.shape[0],)
		assert jnp.allclose(result, expected)

	@allure.title("ExpModule(ZeroMean) gives all ones")
	@allure.description("exp(0) == 1 for any input.")
	def test_exp_zero_mean_gives_ones(self):
		wrapped = ExpModule(ZeroMean())
		x = jnp.array([[1.0], [2.0], [3.0]])
		result = wrapped(x)
		assert jnp.allclose(result, jnp.ones(x.shape[0]))


class TestLogModuleWithMean:
	"""Tests for LogModule wrapping a mean function."""

	@allure.title("LogModule wrapping a mean: instantiation")
	@allure.description("Test that LogModule can wrap a mean function.")
	def test_instantiation(self):
		mean = ConstantMean(constant=2.0)
		wrapped = LogModule(mean)
		assert wrapped.inner is mean

	@allure.title("LogModule wrapping ConstantMean: scalar computation")
	@allure.description("Test log(constant) for a scalar input.")
	def test_scalar_computation(self):
		c = float(jnp.e)  # log(e) == 1
		wrapped = LogModule(ConstantMean(constant=c))
		result = wrapped(jnp.array([0.5]))
		assert jnp.allclose(result, jnp.array(1.0), atol=1e-6)

	@allure.title("LogModule wrapping LinearMean: vector computation")
	@allure.description("Test log(slope * x) for a batch of positive inputs.")
	def test_vector_computation(self):
		slope = 2.0
		wrapped = LogModule(LinearMean(slope=slope))
		x = jnp.array([[1.0], [2.0], [3.0]])
		result = wrapped(x)
		expected = jnp.log(slope * x.squeeze(-1))
		assert result.shape == (x.shape[0],)
		assert jnp.allclose(result, expected)

	@allure.title("LogModule and ExpModule are inverses")
	@allure.description("Log(Exp(mean)) should return the original mean values.")
	def test_log_exp_inverse(self):
		c = 3.0
		wrapped = LogModule(ExpModule(ConstantMean(constant=c)))
		x = jnp.array([1.0])
		result = wrapped(x)
		assert jnp.allclose(result, jnp.array(c), atol=1e-6)


class TestActiveDimsModuleWithMean:
	"""Tests for ActiveDimsModule wrapping a mean function."""

	@allure.title("ActiveDimsModule wrapping a mean: instantiation")
	@allure.description("Test that ActiveDimsModule can wrap a mean function.")
	def test_instantiation(self):
		mean = LinearMean(slope=1.0)
		wrapped = ActiveDimsModule(mean, active_dims=jnp.array([0]))
		assert wrapped.inner is mean

	@allure.title("ActiveDimsModule selects dimensions for a batch mean")
	@allure.description(
		"Test that only selected input dimensions are passed to the mean, "
		"leaving other dimensions irrelevant."
	)
	def test_dimension_selection(self):
		inner_mean = LinearMean(slope=1.0)
		wrapped = ActiveDimsModule(inner_mean, active_dims=jnp.array([0]))

		# 2D batch input: vary dim 0, keep dim 1 (irrelevant) constant at 99
		x = jnp.array([[1.0, 99.0], [2.0, 99.0], [3.0, 99.0]])
		result = wrapped(x)

		# Equivalent to applying inner_mean on x[:, :1]
		expected = inner_mean(x[:, :1])
		assert jnp.allclose(result, expected)

	@allure.title("ActiveDimsModule: non-first dimension selection")
	@allure.description("Test selecting the second dimension from a 3-feature input.")
	def test_non_first_dimension(self):
		inner_mean = LinearMean(slope=2.0)
		wrapped = ActiveDimsModule(inner_mean, active_dims=jnp.array([1]))

		x = jnp.array([[0.0, 3.0, 99.0], [1.0, 4.0, 99.0]])
		result = wrapped(x)

		# mean(x[:, 1:2]) = slope * x[:, 1] = 2 * [3, 4] = [6, 8]
		expected = jnp.array([6.0, 8.0])
		assert jnp.allclose(result, expected)

	@allure.title("ActiveDimsModule: ConstantMean ignores active dims")
	@allure.description("A ConstantMean wrapped with ActiveDims still returns its constant.")
	def test_constant_mean_ignores_dims(self):
		wrapped = ActiveDimsModule(ConstantMean(constant=7.0), active_dims=jnp.array([0]))
		x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
		result = wrapped(x)
		assert jnp.allclose(result, jnp.full(2, 7.0))


class TestWrapperMeanCombinations:
	"""Tests for combinations of wrapper modules on means."""

	@allure.title("Nested wrappers: Exp(Log(mean)) == mean for positive output")
	@allure.description("exp(log(x)) == x for positive x.")
	def test_exp_of_log(self):
		c = 3.0
		wrapped = ExpModule(LogModule(ConstantMean(constant=c)))
		result = wrapped(jnp.array([1.0]))
		assert jnp.allclose(result, jnp.array(c), atol=1e-6)

	@allure.title("ActiveDimsModule wrapping ExpModule wrapping a mean")
	@allure.description("Test that ActiveDims can wrap a transformed mean.")
	def test_active_dims_over_exp(self):
		inner = LinearMean(slope=1.0)
		exp_mean = ExpModule(inner)
		wrapped = ActiveDimsModule(exp_mean, active_dims=jnp.array([0]))

		x = jnp.array([[1.0, 99.0], [2.0, 99.0]])
		result = wrapped(x)

		# Should compute exp(slope * x[:, 0])
		expected = jnp.exp(inner(x[:, :1]))
		assert jnp.allclose(result, expected)

	@allure.title("String representation of wrapped means is valid")
	@allure.description("Test that __str__ produces a non-empty string for wrapped means.")
	def test_str_representation(self):
		assert isinstance(str(ExpModule(LinearMean(slope=1.0))), str)
		assert len(str(ExpModule(LinearMean(slope=1.0)))) > 0
		assert isinstance(str(LogModule(ConstantMean(constant=2.0))), str)
		assert isinstance(str(ActiveDimsModule(LinearMean(slope=1.0), active_dims=jnp.array([0]))), str)


class TestBatchModuleWithMean:
	"""Tests for BatchModule wrapping a mean function — all four batching scenarios."""

	@allure.title("BatchModule with mean: batch over hyperparameters only")
	@allure.description(
		"batch_in_axes=0, batch_over_inputs=False: same inputs for all batch elements, "
		"each element has its own hyperparameters."
	)
	def test_batch_over_hyperparameters(self):
		batch_size = 3
		batch_mean = BatchModule(ConstantMean(constant=1.0), batch_size=batch_size, batch_in_axes=0, batch_over_inputs=False)

		x = jnp.array([[0.0], [1.0], [2.0]])  # shape (3, 1)
		result = batch_mean(x)

		# Output: (B, N) — B mean vectors, one per batch element
		assert result.shape == (batch_size, x.shape[0])
		assert jnp.all(jnp.isfinite(result))

	@allure.title("BatchModule with mean: batch over inputs only")
	@allure.description(
		"batch_in_axes=None, batch_over_inputs=True: distinct inputs per batch element, "
		"shared hyperparameters."
	)
	def test_batch_over_inputs(self):
		batch_size = 4
		batch_mean = BatchModule(LinearMean(slope=2.0), batch_size=batch_size, batch_in_axes=None, batch_over_inputs=True)

		x_batched = jnp.ones((batch_size, 3, 1))  # shape (B, N, D)
		result = batch_mean(x_batched)

		assert result.shape == (batch_size, 3)
		assert jnp.all(jnp.isfinite(result))
		# All batches use the same mean → all outputs should be identical
		for i in range(1, batch_size):
			assert jnp.allclose(result[i], result[0])

	@allure.title("BatchModule with mean: batch over hyperparameters and inputs")
	@allure.description(
		"batch_in_axes=0, batch_over_inputs=True: distinct inputs and distinct hyperparameters "
		"per batch element."
	)
	def test_batch_over_inputs_and_hyperparameters(self):
		batch_size = 3
		batch_mean = BatchModule(LinearMean(slope=1.0), batch_size=batch_size, batch_in_axes=0, batch_over_inputs=True)

		x_batched = jnp.ones((batch_size, 4, 1))  # shape (B, N, D)
		result = batch_mean(x_batched)

		assert result.shape == (batch_size, 4)
		assert jnp.all(jnp.isfinite(result))

	@allure.title("BatchModule with mean: shared hyperparameters and shared inputs")
	@allure.description(
		"batch_in_axes=None, batch_over_inputs=False: all batch elements share the same "
		"hyperparameters and inputs — output is B identical mean vectors."
	)
	def test_shared_hyperparameters_shared_inputs(self):
		batch_size = 3
		mean = ConstantMean(constant=5.0)
		batch_mean = BatchModule(mean, batch_size=batch_size, batch_in_axes=None, batch_over_inputs=False)

		x = jnp.array([[1.0], [2.0], [3.0]])
		result = batch_mean(x)

		assert result.shape == (batch_size, x.shape[0])
		assert jnp.all(jnp.isfinite(result))
		# All batch elements must be identical (same HPs + same inputs)
		expected = mean(x)
		for i in range(batch_size):
			assert jnp.allclose(result[i], expected)

	@allure.title("BatchModule with mean: correct values with batched hyperparameters")
	@allure.description(
		"Verify that each batch element produces the correct mean value for its own hyperparameters."
	)
	def test_correct_values_batched_hyperparameters(self):
		# After BatchModule init, inner.constant will have shape (3,) = [1, 1, 1]
		# We'll use replace() to set distinct values per batch element
		batch_size = 3
		batch_mean = BatchModule(ConstantMean(constant=1.0), batch_size=batch_size, batch_in_axes=0, batch_over_inputs=False)
		batch_mean = batch_mean.replace(constant=jnp.array([1.0, 2.0, 3.0]))

		x = jnp.array([[0.0]])  # single point
		result = batch_mean(x)  # shape (3, 1)

		assert jnp.allclose(result[0], jnp.array([1.0]))
		assert jnp.allclose(result[1], jnp.array([2.0]))
		assert jnp.allclose(result[2], jnp.array([3.0]))