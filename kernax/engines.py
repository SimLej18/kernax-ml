from __future__ import annotations

from equinox import filter_jit, error_if
from jax import Array, vmap
from jax.lax import cond
import jax.numpy as jnp
from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from .AbstractKernel import AbstractKernel
	from .stationary.StationaryKernel import StaticStationaryKernel
	from .other.ConstantKernel import ConstantKernel

class ComputationEngine(ABC):
	"""
	Superclass for all computation engines.

	A computation engine defines how to build cross-covariance vectors/matrices given inputs and a
	kernel.
	Most of the time, the `DenseEngine` is the one wanted, as full covariance matrices are computed.
	It is the default engine associated to all kernels.

	But some other engines can be used to compute only specific parts of the covariance matrices,
	like `DiagonalEngine`, which only computes the diagonal of the covariance matrix.

	Some other engines can also be used when treating inputs with specific structures, like the
	`RegularGridEngine`, which exploits the regular grid structure of the inputs to compute the
	covariance matrix more efficiently.

	For maximal efficiency, some engines might require specific constraints on the inputs or the
	kernel. Asserting these constraints at runtime is sometimes costly. This is why most engines
	come in two versions: `Safe` and `Fast`.

	The `Safe` versions either check the constraints at runtime, or adapt their computations to
	always satisfy the constraints.

	The `Fast` versions skip any check, and assume that the user ensured that the constraints
	are satisfied. Failing to do so may lead to incorrect results or runtime errors. However, you
	can still use the `check_constraints` method to verify that the inputs and kernel are compatible.

	An example of this is using a `RegularGridEngine`. This engine is more efficient when
	the inputs are indeed on a regular grid and the kernel is stationary, as the covariance matrix
	is just a repetition of the same vector, offset by some indices.
	However, checking that the inputs are on a regular grid (according to the metric used by the
	kernel) is costly. The `Safe` version of the engine will check this at runtime, while the `Fast`
	version will skip this check.

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

	@classmethod
	@filter_jit
	def check_constraints(cls, kern: AbstractKernel, x1: Array, x2: Array) -> bool:
		"""
		Check if the inputs and kernel are compatible with the computation engine.

		By default, all inputs and kernels are compatible. Specific engines can override this method
		to implement specific checks, using equinox's `error_if` fonction.

		:param kern: kernel instance containing the hyperparameters
		:param x1: vector array (N, )
		:param x2: vector array (M, )
		:return: True if the inputs are compatible, False otherwise
		"""
		return True


class DenseEngine(ComputationEngine):
	"""
	Dense computation engine.

	Computes full covariance matrices between inputs.
	"""
	pass  # Super-class already implements all needed methods


class FastDiagonalEngine(ComputationEngine):
	"""
	Fast Diagonal computation engine. Returns covariance value if inputs are equal, otherwise 0.
	This engine assumes that the two sets of inputs are identical (meaning that for every i, x1[i] == x2[i]).
	This results in a diagonal covariance matrix.

	An additional speedup is possible when computing it on a stationary of constant kernel, as the
	covariance values are all identical. In this case, the diagonal is just a repetition of the same value.
	"""
	@classmethod
	@filter_jit
	def pairwise_cov(cls, kern, x1: Array, x2: Array) -> Array:
		# It is assumed that x1 == x2
		return kern.static_class.pairwise_cov(kern, x1, x2)

	@classmethod
	@filter_jit
	def cross_cov_matrix(cls, kern: AbstractKernel, x1: Array, x2: Array) -> Array:
		"""
		Compute only the diagonal of the kernel covariance matrix between two identical vector arrays.
		It is assumed that for every i, x1[i] == x2[i].

		:param kern: kernel instance containing the hyperparameters
		:param x1: vector array (N, )
		:param x2: vector array (M, )
		:return: vector array (min(N, M), ) containing the diagonal elements
		"""
		# Import here to avoid circular imports
		from .stationary.StationaryKernel import StaticStationaryKernel
		from .other.ConstantKernel import ConstantKernel

		# Check whether kern inherits from StationaryKernel or ConstantKernel for speedup
		if isinstance(kern.static_class, StaticStationaryKernel) or isinstance(kern, ConstantKernel):
			# Single covariance value for all diagonal elements
			return jnp.eye(x1.shape[0]) * cls.pairwise_cov_if_not_nan(kern, x1[0], x2[0])

		# Else, compute all diagonal elements
		return jnp.eye(x1.shape[0]) * vmap(cls.pairwise_cov_if_not_nan, in_axes=(None, 0, 0))(kern, x1, x2)

	@classmethod
	@filter_jit
	def check_constraints(cls, kern: AbstractKernel, x1: Array, x2: Array) -> bool:
		"""
		Check that the two sets of inputs are identical.

		:param kern: kernel instance containing the hyperparameters
		:param x1: vector array (N, )
		:param x2: vector array (M, )
		:return: True if the inputs are compatible, False otherwise
		"""
		# Check that the two sets of inputs are identical
		x1 = error_if(x1, jnp.any(x1 != x2), "Inputs in x1 and x2 do not match")


class SafeDiagonalEngine(ComputationEngine):
	"""
	Diagonal computation engine. Returns covariance value if inputs are equal, otherwise 0.
	This engine doesn't require any constraint on inputs or kernel, but may be inefficient
	if you always compute the cross-covariance between two equivalent sets of inputs. In this case,
	use FastDiagonalEngine instead.

	When computing on the same set of inputs, this results in a diagonal covariance matrix.
	"""
	@classmethod
	@filter_jit
	def pairwise_cov(cls, kern, x1: Array, x2: Array) -> Array:
		return cond(  # type: ignore[no-any-return]
			jnp.all(x1 == x2), lambda _: kern.static_class.pairwise_cov(kern, x1, x2), lambda _: jnp.array(0.0), None
		)

	@classmethod
	@filter_jit
	def cross_cov_matrix(cls, kern: AbstractKernel, x1: Array, x2: Array) -> Array:
		"""
		Compute diagonal covariance matrix with conditional checks for input equality.
		Returns kernel value when x1[i] == x2[i], otherwise 0.

		:param kern: kernel instance containing the hyperparameters
		:param x1: vector array (N, )
		:param x2: vector array (M, )
		:return: diagonal matrix array (N, N) where off-diagonal elements are 0
		"""
		# Import here to avoid circular imports
		from .stationary.StationaryKernel import StaticStationaryKernel
		from .other.ConstantKernel import ConstantKernel

		# For efficiency with same inputs, check if all inputs are identical
		all_same = jnp.all(x1 == x2)

		def compute_full_diagonal():
			# Compute diagonal elements where x1[i] == x2[i], 0 otherwise
			return jnp.eye(x1.shape[0]) * vmap(cls.pairwise_cov, in_axes=(None, 0, 0))(kern, x1, x2)

		def compute_fast_diagonal():
			# All inputs are same, can optimize
			if isinstance(kern.static_class, StaticStationaryKernel) or isinstance(kern, ConstantKernel):
				# Single covariance value for all diagonal elements
				return jnp.eye(x1.shape[0]) * kern.static_class.pairwise_cov(kern, x1[0], x2[0])
			else:
				# Compute all diagonal elements
				return jnp.eye(x1.shape[0]) * vmap(kern.static_class.pairwise_cov, in_axes=(None, 0, 0))(kern, x1, x2)

		return cond(all_same, lambda _: compute_fast_diagonal(), lambda _: compute_full_diagonal(), None)  # type: ignore[no-any-return]


class FastRegularGridEngine(ComputationEngine):
	"""
	Fast Regular Grid computation engine.

	Exploits the regular grid structure of the inputs to compute the covariance matrix more efficiently.
	This engine assumes that the inputs are indeed on a regular grid according to the metric
	used by the kernel. Failing to do so may lead to incorrect results or runtime errors.
	However, you can still use the `check_constraints` method to verify that the inputs and kernel
	are compatible.
	"""
	@classmethod
	@filter_jit
	def cross_cov_matrix(cls, kern: AbstractKernel, x1: Array, x2: Array) -> Array:
		# We create a circulant matrix from the first row
		n = x1.shape[0]
		g = jnp.arange(n)
		vec = kern.static_class.cross_cov_vector_if_not_nan(kern, x1, x2[0])
		return vmap(lambda i: vec[(g + i)%n])(g)


	@classmethod
	@filter_jit
	def check_constraints(cls, kern: AbstractKernel, x1: Array, x2: Array):
		"""
		Check that the inputs are on a regular grid according to the metric used by the kernel.

		Note: we cannot only check that the kernel is stationary, as composition of stationary kernels
		can also be stationary. We perform this check by computing the k+1 diagonal of the covariance matrix
		and verifying that all values are identical.

		:param kern: kernel instance containing the hyperparameters
		:param x1: vector array (N, )
		:param x2: vector array (M, )
		:return: True if the inputs are compatible, False otherwise
		"""
		# Check that the inputs are on a regular grid according to kern.distance_func, meaning that
		# the distances between consecutive points are all identical
		diag_covs = vmap(kern.static_class.pairwise_cov, in_axes=(None, 0, 0))(kern, x1[:-1], x2[1:])
		error_if(diag_covs, jnp.any(diag_covs != diag_covs[0]),
		         "Inputs are not on a regular grid or the kernel is not stationary.")


class SafeRegularGridEngine(FastRegularGridEngine):
	"""
	Safe Regular Grid computation engine.

	Exploits the regular grid structure of the inputs to compute the covariance matrix more efficiently.
	This engine checks at runtime that the inputs are indeed on a regular grid according to the metric
	used by the kernel.

	Note: at this time, no significant slowdown is observed when using this safe version compared to the fast version.
	"""
	@classmethod
	@filter_jit
	def cross_cov_matrix(cls, kern: AbstractKernel, x1: Array, x2: Array) -> Array:
		# Verify constraints
		cls.check_constraints(kern, x1, x2)

		# Call parent method
		return super().cross_cov_matrix(kern, x1, x2)
