import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import Array, jit

from .WrapperKernel import WrapperKernel


class ARDKernel(WrapperKernel):
	"""
	Wrapper kernel to apply Automatic Relevance Determination (ARD) to the inputs before passing them to the inner kernel.
	Each input dimension is scaled by a separate length scale hyperparameter.
	"""

	length_scales: Array = eqx.field(converter=jnp.asarray)

	def _freeze_inner_lengthscales(self, kernel):
		def map_func(path, leaf):
			if len(path) > 0:
				last_node = path[-1]  # To retrieve the attribute name
				if isinstance(last_node, jtu.GetAttrKey) and last_node.name == "length_scale":
					# Force length scale of 1
					return jnp.ones_like(leaf)
			return leaf

		return jtu.tree_map_with_path(map_func, kernel)

	def __init__(self, inner_kernel, length_scales):
		"""
		:param inner_kernel: the kernel to wrap, must be an instance of AbstractKernel
		:param length_scales: the length scales for each input dimension (1D array of floats)
		"""
		super().__init__(inner_kernel=inner_kernel)
		self.length_scales = length_scales

	@jit
	def __call__(self, x1: Array, x2: None | Array = None) -> Array:
		# TODO: add runtime error if length_scales doesn't match input dimensions
		if x2 is None:
			x2 = x1

		# FIXME: if used in an optimisation setting, inner length_scales can still have other values.
		#  Freezing the inner at every call may be costly/slow down learning.
		#  We should find a proper way to freeze these parameters.
		return self._freeze_inner_lengthscales(self.inner_kernel)(  # type: ignore[no-any-return]
			x1 / self.length_scales, x2 / self.length_scales
		)
