from ..wrappers import DiagKernel, StaticDiagKernel
from .ConstantKernel import ConstantKernel


class WhiteNoiseKernel(DiagKernel):
	"""
	Kernel that returns a value only if the inputs are equal, otherwise returns 0.
	This results in a diagonal cross-covariance matrix.
	"""

	static_class = StaticDiagKernel

	def __init__(self, noise=None, **kwargs):
		super().__init__(inner_kernel=ConstantKernel(noise), **kwargs)

	def __str__(self):
		return f"WhiteNoise({self.inner_kernel})"
