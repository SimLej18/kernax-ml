import equinox as eqx
from equinox import filter_jit
from jax import Array
from jax import numpy as jnp

from ..AbstractKernel import AbstractKernel
from .DotProductKernel import StaticDotProductKernel


class StaticAffineKernel(StaticDotProductKernel):
	@classmethod
	@filter_jit
	def pairwise_cov(cls, kern: AbstractKernel, x1: Array, x2: Array) -> Array:
		"""
		Compute the affine kernel covariance value between two vectors.

		:param kern: the kernel to use, containing hyperparameters (slope_var, offset).
		:param x1: scalar array.
		:param x2: scalar array.
		:return: scalar array (covariance value).
		"""
		# Compute the dot product of the shifted vectors
		return kern.slope_var * cls.distance_func(x1 - kern.offset, x2 - kern.offset)


class AffineKernel(AbstractKernel):
	"""
	Affine Kernel, corresponding to the formula:
	k(x, x') = slope_var * (x - offset).T @ (x' - offset)

	Note: In GPs, samples/posteriors from this kernel will always cross the points (offset, 0).
	If you want to add uncertainty as to where the crossing point is, you should add a
	ConstantKernel to the AffineKernel (i.e. use a SumKernel) where the constant value
	represents the variance at the crossing point.
	"""
	slope_var: Array = eqx.field(converter=jnp.asarray)
	offset: Array = eqx.field(converter=jnp.asarray)
	static_class = StaticAffineKernel

	def __init__(self, slope_var, offset, **kwargs):
		"""
		Initialize the Linear kernel.

		Args:
			slope_var: Weight variance. Controls the slope. Must be non-negative.
			offset: Input offset. Determines the crossing point.
		"""
		# Initialize parent
		super().__init__(**kwargs)

		# Store parameters as-is (no transformation)
		# Variance and offset can be 0, which is incompatible with log-based transforms
		self.slope_var = jnp.asarray(slope_var)
		self.offset = jnp.asarray(offset)

		# Check non-negativity for slope_var
		_ = eqx.error_if(
			self.slope_var, jnp.any(self.slope_var < 0), "slope_var must be non-negative."
		)
