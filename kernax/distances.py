""" Defines distances/comparisons between input vectors, used in different families of kernels. """

import jax.numpy as jnp
from jax import jit, Array

@jit
def euclidean_distance(x1: Array, x2: Array) -> Array:
	"""
	Compute the Euclidean distance between two vectors.

	:param x1: First input vector.
	:param x2: Second input vector.
	:return: Euclidean distance between x1 and x2.
	"""
	return jnp.linalg.norm(x1 - x2)

@jit
def squared_euclidean_distance(x1: Array, x2: Array) -> Array:
	"""
	Compute the squared Euclidean distance between two vectors.

	:param x1: First input vector.
	:param x2: Second input vector.
	:return: Squared Euclidean distance between x1 and x2.
	"""
	return jnp.sum((x1 - x2)**2)

@jit
def manhattan_distance(x1: Array, x2: Array) -> Array:
	"""
	Compute the Manhattan (L1) distance between two vectors.

	:param x1: First input vector.
	:param x2: Second input vector.
	:return: Manhattan distance between x1 and x2.
	"""
	return jnp.sum(jnp.abs(x1 - x2))

@jit
def chebyshev_distance(x1: Array, x2: Array) -> Array:
	"""
	Compute the Chebyshev (L∞) distance between two vectors.

	:param x1: First input vector.
	:param x2: Second input vector.
	:return: Chebyshev distance between x1 and x2.
	"""
	return jnp.max(jnp.abs(x1 - x2))

@jit
def cosine_distance(x1: Array, x2: Array) -> Array:
	"""
	Compute the Cosine distance between two vectors.

	:param x1: First input vector.
	:param x2: Second input vector.
	:return: Cosine distance between x1 and x2.
	"""
	dot_product = jnp.dot(x1, x2)
	norm_x1 = jnp.linalg.norm(x1)
	norm_x2 = jnp.linalg.norm(x2)
	return 1.0 - (dot_product / (norm_x1 * norm_x2))

@jit
def minkowski_distance(x1: Array, x2: Array, p: float) -> Array:
	"""
	Compute the Minkowski distance between two vectors.

	:param x1: First input vector.
	:param x2: Second input vector.
	:param p: Order of the norm (p >= 1).
	:return: Minkowski distance between x1 and x2.
	"""
	return jnp.sum(jnp.abs(x1 - x2) ** p) ** (1 / p)

@jit
def hamming_distance(x1: Array, x2: Array) -> Array:
	"""
	Compute the Hamming distance between two vectors.

	:param x1: First input vector (binary or categorical).
	:param x2: Second input vector (binary or categorical).
	:return: Hamming distance between x1 and x2.
	"""
	return jnp.sum(x1 != x2)

@jit
def jaccard_distance(x1: Array, x2: Array) -> Array:
	"""
	Compute the Jaccard distance between two binary vectors.

	:param x1: First input vector (binary).
	:param x2: Second input vector (binary).
	:return: Jaccard distance between x1 and x2.
	"""
	intersection = jnp.sum(jnp.minimum(x1, x2))
	union = jnp.sum(jnp.maximum(x1, x2))
	return 1.0 - (intersection / union)

@jit
def mahalanobis_distance(x1: Array, x2: Array, inv_cov_matrix: Array) -> Array:
	"""
	Compute the Mahalanobis distance between two vectors.

	:param x1: First input vector.
	:param x2: Second input vector.
	:param inv_cov_matrix: Inverse of the covariance matrix.
	:return: Mahalanobis distance between x1 and x2.
	"""
	diff = x1 - x2
	return jnp.sqrt(jnp.dot(jnp.dot(diff.T, inv_cov_matrix), diff))

@jit
def canberra_distance(x1: Array, x2: Array) -> Array:
	"""
	Compute the Canberra distance between two vectors.

	:param x1: First input vector.
	:param x2: Second input vector.
	:return: Canberra distance between x1 and x2.
	"""
	return jnp.sum(jnp.abs(x1 - x2) / (jnp.abs(x1) + jnp.abs(x2) + 1e-10))  # Added small constant to avoid division by zero

@jit
def bray_curtis_distance(x1: Array, x2: Array) -> Array:
	"""
	Compute the Bray-Curtis distance between two vectors.

	:param x1: First input vector.
	:param x2: Second input vector.
	:return: Bray-Curtis distance between x1 and x2.
	"""
	return jnp.sum(jnp.abs(x1 - x2)) / jnp.sum(jnp.abs(x1 + x2) + 1e-10)  # Added small constant to avoid division by zero

@jit
def correlation_distance(x1: Array, x2: Array) -> Array:
	"""
	Compute the Correlation distance between two vectors.

	:param x1: First input vector.
	:param x2: Second input vector.
	:return: Correlation distance between x1 and x2.
	"""
	x1_mean = jnp.mean(x1)
	x2_mean = jnp.mean(x2)
	numerator = jnp.sum((x1 - x1_mean) * (x2 - x2_mean))
	denominator = jnp.sqrt(jnp.sum((x1 - x1_mean) ** 2) * jnp.sum((x2 - x2_mean) ** 2))
	return 1.0 - (numerator / (denominator + 1e-10))  # Added small constant to avoid division by zero

@jit
def dot_product(x1: Array, x2: Array) -> Array:
	"""
	Compute the Dot Product distance between two vectors.

	:param x1: First input vector.
	:param x2: Second input vector.
	:return: Dot Product between x1 and x2.
	"""
	return x1.T @ x2
