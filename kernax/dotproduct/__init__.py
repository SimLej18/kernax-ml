"""Dot-product based kernels."""

from .LinearKernel import LinearKernel, StaticLinearKernel
from .PolynomialKernel import PolynomialKernel, StaticPolynomialKernel

__all__ = [
	"LinearKernel",
	"StaticLinearKernel",
	"PolynomialKernel",
	"StaticPolynomialKernel",
]
