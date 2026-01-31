import equinox as eqx
import jax.numpy as jnp
from equinox import filter_jit
from jax import Array

from ..AbstractKernel import AbstractKernel, StaticAbstractKernel
from ..utils import format_jax_array


class StaticConstantKernel(StaticAbstractKernel):
	@classmethod
	@filter_jit
	def pairwise_cov(cls, kern: AbstractKernel, x1: Array, x2: Array) -> Array:
		"""
		Compute the constant kernel covariance value.

		:param kern: the kernel to use, containing hyperparameters
		:param x1: scalar array (ignored)
		:param x2: scalar array (ignored)
		:return: scalar array (constant value)
		"""
		kern = eqx.combine(kern)
		return kern.value  # type: ignore[no-any-return,attr-defined]


class ConstantKernel(AbstractKernel):
	"""
	Constant Kernel

	Returns a constant value regardless of inputs.
	The value parameter can be any real number (positive or negative).
	No transformation is applied since there's no positivity constraint.
	"""

	value: Array = eqx.field(converter=jnp.asarray)
	static_class = StaticConstantKernel

	def __init__(self, value=1.0):
		"""
		Initialize the Constant kernel.

		Args:
			value: The constant value to return (can be any real number)
		"""
		super().__init__()
		self.value = jnp.asarray(value)

	def __str__(self):
		return format_jax_array(self.value)
