from functools import partial

import jax.numpy as jnp
from jax import Array, jit
from jax.lax import cond

from ..AbstractKernel import StaticAbstractKernel
from .WrapperKernel import WrapperKernel


class StaticDiagKernel(StaticAbstractKernel):
	"""
	Static kernel that returns a value only if the inputs are equal, otherwise returns 0.
	This results in a diagonal cross-covariance matrix.
	"""

	@classmethod
	@partial(jit, static_argnums=(0,))
	def pairwise_cov(cls, kern, x1: Array, x2: Array) -> Array:
		return cond(  # type: ignore[no-any-return]
			jnp.all(x1 == x2), lambda _: kern.inner_kernel(x1, x2), lambda _: jnp.array(0.0), None
		)


class DiagKernel(WrapperKernel):
	"""
	Kernel that returns a value only if the inputs are equal, otherwise returns 0.
	This results in a diagonal cross-covariance matrix.
	"""

	static_class = StaticDiagKernel

	def __init__(self, inner_kernel=None):
		super().__init__(inner_kernel=inner_kernel)

	def __str__(self):
		return f"Diag({self.inner_kernel})"
