"""
Pytest configuration and fixtures for Kernax tests.
"""

import jax
import pytest


@pytest.fixture
def random_key():
	"""Provides a JAX random key for tests."""
	return jax.random.PRNGKey(0)


@pytest.fixture
def sample_1d_data(random_key):
	"""Provides sample 1D data for testing."""
	key1, key2 = jax.random.split(random_key)
	x1 = jax.random.uniform(key1, (10, 1))
	x2 = jax.random.uniform(key2, (15, 1))
	return x1, x2


@pytest.fixture
def sample_2d_data(random_key):
	"""Provides sample 2D data for testing."""
	key1, key2 = jax.random.split(random_key)
	x1 = jax.random.uniform(key1, (10, 2))
	x2 = jax.random.uniform(key2, (15, 2))
	return x1, x2


@pytest.fixture
def sample_batched_data(random_key):
	"""Provides sample batched data for testing."""
	key1, key2 = jax.random.split(random_key)
	x1 = jax.random.uniform(key1, (5, 10, 1))
	x2 = jax.random.uniform(key2, (5, 15, 1))
	return x1, x2
