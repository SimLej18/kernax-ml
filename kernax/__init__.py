"""
kernax-ml: A JAX-based kernel library for Gaussian Processes.

kernax-ml provides a collection of kernel functions (covariance functions) for
Gaussian Process models, with support for automatic differentiation, JIT
compilation, and composable kernel operations.
"""

__version__ = "0.6.1-alpha"
__author__ = "S. Lejoly"
__email__ = "simon.lejoly@unamur.be"
__license__ = "MIT"

# Import transformation utilities
from .module import AbstractModule
from .AbstractKernel import AbstractKernel
from .AbstractMean import AbstractMean

# Import dot-product kernels
from .dotproduct import (
	LinearKernel,
	AffineKernel,
	PolynomialKernel,
	SigmoidKernel,
)

# Import operator modules
from .operators import (
	AbstractOperatorModule,
	ProductModule,
	SumModule,
)

# Import other kernels
from .other import (
	ConstantKernel,
	VarianceKernel,
)

# Import stationary kernels
from .stationary import (
	Matern12Kernel,
	Matern32Kernel,
	Matern52Kernel,
	PeriodicKernel,
	RationalQuadraticKernel,
	RBFKernel,
	SEKernel,
	FeatureKernel,
	WhiteNoiseKernel,
)

# Import wrapper modules/kernels
from .wrappers import (
	AbstractWrapperModule,
	ActiveDimsModule,
	ARDKernel,
	BatchModule,
	BlockDiagKernel,
	BlockKernel,
	ExpModule,
	LogModule,
	NegModule,
	InputSpecificParamModule,
)

# Import mean functions
from .means import (
	AffineMean,
	ConstantMean,
	LinearMean,
	ZeroMean,
)

__all__ = [
	# Package metadata
	"__version__",
	"__author__",
	"__email__",
	"__license__",
	# Base classes
	"AbstractModule",
	"AbstractKernel",
	"AbstractMean",
	# Base kernels
	"SEKernel",
	"RBFKernel",
	"ConstantKernel",
	"LinearKernel",
	"AffineKernel",
	"PeriodicKernel",
	"RationalQuadraticKernel",
	"WhiteNoiseKernel",
	"PolynomialKernel",
	"SigmoidKernel",
	"VarianceKernel",
	"FeatureKernel",
	# Matern family
	"Matern12Kernel",
	"Matern32Kernel",
	"Matern52Kernel",
	# Operator modules
	"AbstractOperatorModule",
	"SumModule",
	"ProductModule",
	# Wrapper modules/kernels
	"AbstractWrapperModule",
	"NegModule",
	"ExpModule",
	"LogModule",
	"ActiveDimsModule",
	"ARDKernel",
	"BatchModule",
	"BlockKernel",
	"BlockDiagKernel",
	"InputSpecificParamModule",
	# Mean functions
	"ZeroMean",
	"ConstantMean",
	"LinearMean",
	"AffineMean",
]
