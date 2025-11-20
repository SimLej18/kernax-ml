"""
Tests for batched operations and distinct hyperparameters.
"""

import pytest
import jax
import jax.numpy as jnp
from Kernax import RBFKernel, LinearKernel, Matern32Kernel


class TestBatchedOperations:
    """Tests for batched kernel computations."""

    def test_shared_hyperparameters(self, sample_batched_data):
        """Test batched computation with shared hyperparameters."""
        kernel = RBFKernel(length_scale=1.0, variance=1.0)
        x1, x2 = sample_batched_data

        result = kernel(x1, x2)

        batch_size = x1.shape[0]
        n_points_1 = x1.shape[1]
        n_points_2 = x2.shape[1]

        assert result.shape == (batch_size, n_points_1, n_points_2)
        assert jnp.all(jnp.isfinite(result))

    def test_distinct_hyperparameters(self, random_key):
        """Test batched computation with distinct hyperparameters per batch."""
        batch_size = 10
        n_points = 5

        # Create batched data
        key1, key2 = jax.random.split(random_key)
        x1 = jax.random.uniform(key1, (batch_size, n_points, 1))
        x2 = jax.random.uniform(key2, (batch_size, n_points, 1))

        # Create distinct hyperparameters for each batch
        length_scales = jnp.linspace(0.5, 2.0, batch_size)
        variances = jnp.linspace(0.5, 1.5, batch_size)

        kernel = RBFKernel(length_scale=length_scales, variance=variances)
        result = kernel(x1, x2)

        assert result.shape == (batch_size, n_points, n_points)
        assert jnp.all(jnp.isfinite(result))

    def test_distinct_hyperparameters_different_values(self, random_key):
        """Test that distinct hyperparameters produce different results."""
        batch_size = 5
        n_points = 3

        # Create same data for all batches
        key = random_key
        x = jax.random.uniform(key, (batch_size, n_points, 1))

        # Use distinct hyperparameters
        length_scales = jnp.array([0.5, 1.0, 1.5, 2.0, 2.5])
        kernel = RBFKernel(length_scale=length_scales, variance=1.0)
        result = kernel(x, x)

        # Results for different batches should be different
        # (except diagonal which should always be variance)
        for i in range(batch_size - 1):
            # Compare off-diagonal elements
            assert not jnp.allclose(result[i], result[i + 1])

    def test_has_distinct_hyperparameters(self):
        """Test detection of distinct hyperparameters."""
        # Shared hyperparameters
        kernel_shared = RBFKernel(length_scale=1.0, variance=1.0)
        assert not kernel_shared.has_distinct_hyperparameters(10)

        # Distinct hyperparameters
        length_scales = jnp.linspace(0.5, 2.0, 10)
        kernel_distinct = RBFKernel(length_scale=length_scales, variance=1.0)
        assert kernel_distinct.has_distinct_hyperparameters(10)

    def test_multiple_kernel_types_batched(self, random_key):
        """Test that different kernel types work with batched operations."""
        batch_size = 5
        n_points = 5

        key1, key2 = jax.random.split(random_key)
        x1 = jax.random.uniform(key1, (batch_size, n_points, 1))
        x2 = jax.random.uniform(key2, (batch_size, n_points, 1))

        length_scales = jnp.linspace(0.5, 2.0, batch_size)

        # Test different kernel types
        kernels = [
            RBFKernel(length_scale=length_scales, variance=1.0),
            Matern32Kernel(length_scale=length_scales),
        ]

        for kernel in kernels:
            result = kernel(x1, x2)
            assert result.shape == (batch_size, n_points, n_points)
            assert jnp.all(jnp.isfinite(result))


class TestNaNHandling:
    """Tests for NaN-aware computations."""

    def test_nan_in_input(self):
        """Test that NaN in input produces NaN in output."""
        kernel = RBFKernel(length_scale=1.0, variance=1.0)
        x1 = jnp.array([1.0, jnp.nan, 3.0])
        x2 = jnp.array([1.5, 2.5, 3.5])

        result = kernel(x1, x2)

        # Row corresponding to NaN input should be NaN
        assert jnp.all(jnp.isnan(result[1, :]))

    def test_nan_propagation_in_matrix(self):
        """Test NaN propagation in matrix computations."""
        kernel = RBFKernel(length_scale=1.0, variance=1.0)

        # Create matrix with some NaN values
        x = jnp.array([[1.0], [jnp.nan], [3.0], [4.0]])

        result = kernel(x, x)

        # Row and column corresponding to NaN should be NaN
        assert jnp.all(jnp.isnan(result[1, :]))
        assert jnp.all(jnp.isnan(result[:, 1]))

        # Other elements should be finite
        valid_mask = jnp.ones((4, 4), dtype=bool)
        valid_mask[1, :] = False
        valid_mask[:, 1] = False
        assert jnp.all(jnp.isfinite(result[valid_mask]))

    def test_padded_data(self, random_key):
        """Test computation with padded (NaN-filled) data."""
        # Simulate padded sequences of different lengths
        max_len = 10
        actual_lengths = [5, 7, 10, 6]
        batch_size = len(actual_lengths)

        # Create padded data
        key = random_key
        x = jax.random.uniform(key, (batch_size, max_len, 1))

        # Add NaN padding
        for i, length in enumerate(actual_lengths):
            x = x.at[i, length:, :].set(jnp.nan)

        kernel = RBFKernel(length_scale=1.0, variance=1.0)
        result = kernel(x, x)

        assert result.shape == (batch_size, max_len, max_len)

        # Check that valid regions have finite values
        for i, length in enumerate(actual_lengths):
            valid_region = result[i, :length, :length]
            assert jnp.all(jnp.isfinite(valid_region))


class TestDimensionHandling:
    """Tests for automatic dimension handling."""

    def test_scalar_to_scalar(self):
        """Test scalar x scalar -> scalar."""
        kernel = RBFKernel(length_scale=1.0, variance=1.0)
        x1 = jnp.array([1.0])
        x2 = jnp.array([2.0])
        result = kernel(x1, x2)
        assert result.shape == ()

    def test_vector_to_scalar(self):
        """Test vector x scalar -> vector."""
        kernel = RBFKernel(length_scale=1.0, variance=1.0)
        x1 = jnp.array([[1.0], [2.0], [3.0]])
        x2 = jnp.array([2.0])
        result = kernel(x1, x2)
        assert result.shape == (3,)

    def test_vector_to_vector(self):
        """Test vector x vector -> matrix."""
        kernel = RBFKernel(length_scale=1.0, variance=1.0)
        x1 = jnp.array([[1.0], [2.0], [3.0]])
        x2 = jnp.array([[1.5], [2.5]])
        result = kernel(x1, x2)
        assert result.shape == (3, 2)

    def test_batch_to_batch(self):
        """Test batch x batch -> batch of matrices."""
        kernel = RBFKernel(length_scale=1.0, variance=1.0)
        x1 = jnp.ones((5, 3, 1))
        x2 = jnp.ones((5, 4, 1))
        result = kernel(x1, x2)
        assert result.shape == (5, 3, 4)

    def test_dimension_mismatch_error(self):
        """Test that dimension mismatch raises error."""
        kernel = RBFKernel(length_scale=1.0, variance=1.0)
        x1 = jnp.ones((5, 3, 1))  # Batch of 5
        x2 = jnp.ones((3, 4, 1))  # Batch of 3

        with pytest.raises(ValueError):
            kernel(x1, x2)