"""
Tests for batched operations and distinct hyperparameters.

NOTE: These tests use the old API with batched hyperparameters directly in kernels.
This functionality has been replaced by BatchKernel wrapper.
These tests are kept for legacy compatibility but marked as skipped.
Use BatchKernel for new code - see tests/test_wrapper_kernels.py for examples.
"""

import pytest
import jax
import jax.numpy as jnp
from kernax import SEKernel, LinearKernel, Matern32Kernel, BatchKernel


class TestBatchedOperations:
    """Tests for batched kernel computations - LEGACY, use BatchKernel instead."""

    @pytest.mark.skip(reason="Legacy test - batched inputs now require BatchKernel wrapper")
    def test_shared_hyperparameters(self, sample_batched_data):
        """Test batched computation with shared hyperparameters."""
        kernel = SEKernel(length_scale=1.0)
        x1, x2 = sample_batched_data

        result = kernel(x1, x2)

        batch_size = x1.shape[0]
        n_points_1 = x1.shape[1]
        n_points_2 = x2.shape[1]

        assert result.shape == (batch_size, n_points_1, n_points_2)
        assert jnp.all(jnp.isfinite(result))

    @pytest.mark.skip(reason="Legacy test - use BatchKernel for distinct hyperparameters")
    def test_distinct_hyperparameters(self, random_key):
        """Test batched computation with distinct hyperparameters per batch."""
        pass

    @pytest.mark.skip(reason="Legacy test - use BatchKernel for distinct hyperparameters")
    def test_distinct_hyperparameters_different_values(self, random_key):
        """Test that distinct hyperparameters produce different results."""
        pass

    @pytest.mark.skip(reason="Legacy test - has_distinct_hyperparameters removed from API")
    def test_has_distinct_hyperparameters(self):
        """Test detection of distinct hyperparameters."""
        pass

    @pytest.mark.skip(reason="Legacy test - use BatchKernel for batched operations")
    def test_multiple_kernel_types_batched(self, random_key):
        """Test that different kernel types work with batched operations."""
        pass


class TestNaNHandling:
    """Tests for NaN-aware computations."""

    def test_nan_in_input(self):
        """Test that NaN in input produces NaN in output."""
        kernel = SEKernel(length_scale=1.0)
        x1 = jnp.array([[1.0], [jnp.nan], [3.0]])  # Shape (3, 1) - 2D matrix
        x2 = jnp.array([[1.5], [2.5], [3.5]])     # Shape (3, 1)

        result = kernel(x1, x2)  # Shape (3, 3)

        # Row corresponding to NaN input should be NaN
        assert jnp.all(jnp.isnan(result[1, :]))

    def test_nan_propagation_in_matrix(self):
        """Test NaN propagation in matrix computations."""
        kernel = SEKernel(length_scale=1.0)

        # Create matrix with some NaN values
        x = jnp.array([[1.0], [jnp.nan], [3.0], [4.0]])

        result = kernel(x, x)

        # Row and column corresponding to NaN should be NaN
        assert jnp.all(jnp.isnan(result[1, :]))
        assert jnp.all(jnp.isnan(result[:, 1]))

        # Other elements should be finite - use JAX immutable assignment
        valid_mask = jnp.ones((4, 4), dtype=bool)
        valid_mask = valid_mask.at[1, :].set(False)
        valid_mask = valid_mask.at[:, 1].set(False)
        assert jnp.all(jnp.isfinite(result[valid_mask]))

    @pytest.mark.skip(reason="Legacy test - batched NaN handling now requires BatchKernel")
    def test_padded_data(self, random_key):
        """Test computation with padded (NaN-filled) data."""
        pass


class TestDimensionHandling:
    """Tests for automatic dimension handling."""

    def test_scalar_to_scalar(self):
        """Test scalar x scalar -> scalar."""
        kernel = SEKernel(length_scale=1.0)
        x1 = jnp.array([1.0])
        x2 = jnp.array([2.0])
        result = kernel(x1, x2)
        assert result.shape == ()

    def test_vector_to_scalar(self):
        """Test vector x scalar -> vector."""
        kernel = SEKernel(length_scale=1.0)
        x1 = jnp.array([[1.0], [2.0], [3.0]])
        x2 = jnp.array([2.0])
        result = kernel(x1, x2)
        assert result.shape == (3,)

    def test_vector_to_vector(self):
        """Test vector x vector -> matrix."""
        kernel = SEKernel(length_scale=1.0)
        x1 = jnp.array([[1.0], [2.0], [3.0]])
        x2 = jnp.array([[1.5], [2.5]])
        result = kernel(x1, x2)
        assert result.shape == (3, 2)

    @pytest.mark.skip(reason="Batch operations now require BatchKernel wrapper")
    def test_batch_to_batch(self):
        """Test batch x batch -> batch of matrices."""
        pass

    @pytest.mark.skip(reason="Dimension mismatch behavior changed with BatchKernel")
    def test_dimension_mismatch_error(self):
        """Test that dimension mismatch raises error."""
        pass