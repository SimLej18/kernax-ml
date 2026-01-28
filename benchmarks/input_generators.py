"""
Input generation helpers for kernax-ml benchmarks.

This module provides reusable functions to generate various types of inputs
for benchmarking kernel performance.
"""

import jax
import jax.numpy as jnp
import jax.random as jr


def generate_1d_regular_grid(n_points=10000, min_val=-500, max_val=500):
	"""Generate 1D regular grid inputs."""
	x = jnp.linspace(min_val, max_val, n_points)
	return x.reshape(-1, 1)


def generate_2d_regular_grid(n_points_per_dim=100, min_val=-20, max_val=20):
	"""Generate 2D regular grid inputs."""
	x1 = jnp.linspace(min_val, max_val, n_points_per_dim)
	x2 = jnp.linspace(min_val, max_val, n_points_per_dim)
	x1_grid, x2_grid = jnp.meshgrid(x1, x2, indexing="ij")
	return jnp.stack([x1_grid.ravel(), x2_grid.ravel()], axis=1)


def generate_5d_regular_grid(n_points_per_dim=5, min_val=-2, max_val=2):
	"""Generate 5D regular grid inputs."""
	axes = [jnp.linspace(min_val, max_val, n_points_per_dim) for _ in range(5)]
	grids = jnp.meshgrid(*axes, indexing="ij")
	return jnp.stack([g.ravel() for g in grids], axis=1)


def generate_random_inputs(key, n_points, n_dims, min_val, max_val):
	"""Generate random inputs."""
	x = jr.uniform(key, (n_points, n_dims), minval=min_val, maxval=max_val)
	x.block_until_ready()
	return x


def add_missing_values(key, x, missing_rate=0.25):
	"""Add missing values (NaN) to inputs."""
	n_points = x.shape[0]
	keep_mask = jr.bernoulli(key, 1 - missing_rate, (n_points,))
	x_with_nan = jnp.where(keep_mask[:, None], x, jnp.nan)
	x_with_nan.block_until_ready()
	return x_with_nan


def generate_batched_random_inputs(key, batch_size, n_points, n_dims, min_val, max_val):
	"""Generate batched random inputs using vmap."""
	keys = jr.split(key, batch_size)
	batched_fn = jax.vmap(
		lambda k: jr.uniform(k, (n_points, n_dims), minval=min_val, maxval=max_val)
	)
	x = batched_fn(keys)
	x.block_until_ready()
	return x