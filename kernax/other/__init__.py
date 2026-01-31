"""Other kernel types."""

from .ConstantKernel import ConstantKernel, StaticConstantKernel
from .WhiteNoiseKernel import WhiteNoiseKernel

__all__ = [
	"ConstantKernel",
	"StaticConstantKernel",
	"WhiteNoiseKernel",
]
