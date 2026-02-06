from ..engines import SafeDiagonalEngine
from .ConstantKernel import ConstantKernel, StaticConstantKernel


class WhiteNoiseKernel(ConstantKernel):
	"""
	White noise kernel that returns a constant value only on the diagonal.

	This kernel is equivalent to a ConstantKernel with a SafeDiagonalEngine computation engine.
	It returns the noise value when inputs are equal, and 0 otherwise, resulting in a diagonal
	covariance matrix.
	"""

	static_class = StaticConstantKernel

	def __init__(self, noise=1.0, **kwargs):
		# Set the computation engine to SafeDiagonalEngine for diagonal computation
		kwargs['computation_engine'] = SafeDiagonalEngine
		super().__init__(value=noise, **kwargs)
