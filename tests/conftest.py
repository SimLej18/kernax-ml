"""
Pytest configuration and fixtures for kernax-ml tests.
"""

import jax.random as jr
import pytest


@pytest.fixture
def random_key():
	"""Provides a JAX random key for tests."""
	return jr.PRNGKey(0)


@pytest.fixture
def sample_1d_data(random_key):
	"""Provides sample 1D data for testing."""
	key1, key2 = jr.split(random_key)
	x1 = jr.uniform(key1, (10, 1))
	x2 = jr.uniform(key2, (15, 1))
	return x1, x2


@pytest.fixture
def sample_2d_data(random_key):
	"""Provides sample 2D data for testing."""
	key1, key2 = jr.split(random_key)
	x1 = jr.uniform(key1, (10, 2))
	x2 = jr.uniform(key2, (15, 2))
	return x1, x2


@pytest.fixture
def sample_batched_data(random_key):
	"""Provides sample batched data for testing."""
	key1, key2 = jr.split(random_key)
	x1 = jr.uniform(key1, (5, 10, 1))
	x2 = jr.uniform(key2, (5, 15, 1))
	return x1, x2
