"""
kernax-ml: A JAX-based kernel library for Gaussian Processes.

kernax-ml provides a collection of kernel functions (covariance functions) for
Gaussian Process models, with support for automatic differentiation, JIT
compilation, and composable kernel operations.
"""

__version__ = "0.3.3-alpha"
__author__ = "S. Lejoly"
__email__ = "simon.lejoly@unamur.be"
__license__ = "MIT"

from .AbstractKernel import AbstractKernel, StaticAbstractKernel

# Import operator kernels
from .operators import (
	OperatorKernel,
	ProductKernel,
	SumKernel,
)

# Import base kernels from stationary
from .stationary import (
	ConstantKernel,
	LinearKernel,
	Matern12Kernel,
	Matern32Kernel,
	Matern52Kernel,
	PeriodicKernel,
	PolynomialKernel,
	RationalQuadraticKernel,
	RBFKernel,
	SEKernel,
	StaticConstantKernel,
	StaticLinearKernel,
	StaticMatern12Kernel,
	StaticMatern32Kernel,
	StaticMatern52Kernel,
	StaticPeriodicKernel,
	StaticPolynomialKernel,
	StaticRationalQuadraticKernel,
	StaticSEKernel,
	WhiteNoiseKernel,
)

# Import wrapper kernels
from .wrappers import (
	ActiveDimsKernel,
	ARDKernel,
	BatchKernel,
	BlockDiagKernel,
	BlockKernel,
	DiagKernel,
	ExpKernel,
	LogKernel,
	NegKernel,
	WrapperKernel,
)

__all__ = [
	# Package metadata
	"__version__",
	"__author__",
	"__email__",
	"__license__",
	# Base classes
	"StaticAbstractKernel",
	"AbstractKernel",
	# Base kernels
	"StaticSEKernel",
	"SEKernel",
	"RBFKernel",
	"StaticConstantKernel",
	"ConstantKernel",
	"StaticLinearKernel",
	"LinearKernel",
	"StaticPeriodicKernel",
	"PeriodicKernel",
	"StaticRationalQuadraticKernel",
	"RationalQuadraticKernel",
	"StaticPolynomialKernel",
	"PolynomialKernel",
	"WhiteNoiseKernel",
	# Matern family
	"StaticMatern12Kernel",
	"Matern12Kernel",
	"StaticMatern32Kernel",
	"Matern32Kernel",
	"StaticMatern52Kernel",
	"Matern52Kernel",
	# Composite kernels
	"OperatorKernel",
	"SumKernel",
	"ProductKernel",
	# Wrapper kernels
	"WrapperKernel",
	"NegKernel",
	"ExpKernel",
	"LogKernel",
	"DiagKernel",
	"BatchKernel",
	"ActiveDimsKernel",
	"ARDKernel",
	"BlockKernel",
	"BlockDiagKernel",
]
