from __future__ import annotations

from equinox import filter_jit
from jax import Array
from abc import ABC

class ComputationEngine(ABC):
	"""
	Superclass for all computation engines.

	A computation engine defines how to build cross-covariance vectors/matrices given inputs and a
	kernel.
	Most of the time, the `DenseEngine` is the one wanted, as full covariance matrices are computed.
	It is the default engine associated to kernels.

	But some other engines can be used to compute only specific parts of the covariance matrices,
	like `DiagonalEngine`, which only computes the diagonal of the covariance matrix.

	Some other engines can also be used when treating inputs with specific structures, like the
	`RegularGridEngine`, which exploits the regular grid structure of the inputs to compute the
	covariance matrix more efficiently.

	Computation engines are an interface between the kernels and their abstract implementation.
	They should implement all the methods defined in this superclass.
	"""
	@classmethod
	@filter_jit
	def pairwise_cov(cls, kern: AbstractKernel, x1: Array, x2: Array) -> Array:
		"""
		Compute the kernel covariance value between two vectors.

		:param kern: kernel instance containing the hyperparameters
		:param x1: scalar array
		:param x2: scalar array
		:return: scalar array
		"""
		return kern.static_class.pairwise_cov(kern, x1, x2)

	@classmethod
	@filter_jit
	def pairwise_cov_if_not_nan(cls, kern: AbstractKernel, x1: Array, x2: Array) -> Array:
		"""
		Returns NaN if either x1 or x2 is NaN, otherwise calls the pairwise_cov method.

		:param kern: kernel instance containing the hyperparameters
		:param x1: scalar array
		:param x2: scalar array
		:return: scalar array
		"""
		return kern.static_class.pairwise_cov_if_not_nan(kern, x1, x2)

	@classmethod
	@filter_jit
	def cross_cov_vector(cls, kern: AbstractKernel, x1: Array, x2: Array) -> Array:
		"""
		Compute the kernel cross covariance values between an array of vectors (matrix) and a vector.

		:param kern: kernel instance containing the hyperparameters
		:param x1: vector array (N, )
		:param x2: scalar array
		:return: vector array (N, )
		"""
		return kern.static_class.cross_cov_vector(kern, x1, x2)

	@classmethod
	@filter_jit
	def cross_cov_vector_if_not_nan(
		cls, kern: AbstractKernel, x1: Array, x2: Array, **kwargs
	) -> Array:
		"""
		Returns an array of NaN if scalar is NaN, otherwise calls the cross_cov_vector method.

		:param kern: kernel instance containing the hyperparameters
		:param x1: vector array (N, )
		:param x2: scalar array
		:param kwargs: hyperparameters of the kernel
		:return: vector array (N, )
		"""
		return kern.static_class.cross_cov_vector_if_not_nan(kern, x1, x2, **kwargs)

	@classmethod
	@filter_jit
	def cross_cov_matrix(cls, kern: AbstractKernel, x1: Array, x2: Array) -> Array:
		"""
		Compute the kernel covariance matrix between two vector arrays.

		:param kern: kernel instance containing the hyperparameters
		:param x1: vector array (N, )
		:param x2: vector array (M, )
		:return: matrix array (N, M)
		"""
		return kern.static_class.cross_cov_matrix(kern, x1, x2)


class DenseEngine(ComputationEngine):
	"""
	Dense computation engine.

	Computes full covariance matrices between inputs.
	"""
	pass




