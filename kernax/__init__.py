"""
kernax-ml: A JAX-based kernel library for Gaussian Processes.

kernax-ml provides a collection of kernel functions (covariance functions) for
Gaussian Process models, with support for automatic differentiation, JIT
compilation, and composable kernel operations.
"""

__version__ = "0.5.0-alpha"
__author__ = "S. Lejoly"
__email__ = "simon.lejoly@unamur.be"
__license__ = "MIT"

# Import transformation utilities
from . import transforms
from .module import AbstractModule, StaticAbstractModule
from .AbstractKernel import AbstractKernel, StaticAbstractKernel
from .AbstractMean import AbstractMean, StaticAbstractMean

# Import configuration system
from .config import config

# Import dot-product kernels
from .dotproduct import (
	LinearKernel,
	PolynomialKernel,
	SigmoidKernel,
	StaticLinearKernel,
	StaticPolynomialKernel,
	StaticSigmoidKernel,
)

# Import operator modules
from .operators import (
	OperatorModule,
	ProductModule,
	SumModule,
)

# Import other kernels
from .other import (
	ConstantKernel,
	StaticConstantKernel,
	VarianceKernel,
	WhiteNoiseKernel,
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
	StaticMatern12Kernel,
	StaticMatern32Kernel,
	StaticMatern52Kernel,
	StaticPeriodicKernel,
	StaticRationalQuadraticKernel,
	StaticSEKernel,
	StaticFeatureKernel,
)

# Import wrapper modules/kernels
from .wrappers import (
	ActiveDimsModule,
	ARDKernel,
	BatchModule,
	BlockDiagKernel,
	BlockKernel,
	ExpModule,
	LogModule,
	NegModule,
	WrapperModule,
)

# Import mean functions
from .means import (
	AffineMean,
	ConstantMean,
	LinearMean,
	StaticAffineMean,
	StaticConstantMean,
	StaticLinearMean,
	StaticZeroMean,
	ZeroMean,
)

__all__ = [
	# Package metadata
	"__version__",
	"__author__",
	"__email__",
	"__license__",
	# Configuration
	"config",
	# Transformations
	"transforms",
	# Base classes
	"StaticAbstractModule",
	"AbstractModule",
	"StaticAbstractKernel",
	"AbstractKernel",
	"StaticAbstractMean",
	"AbstractMean",
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
	"StaticSigmoidKernel",
	"SigmoidKernel",
	"VarianceKernel",
	"WhiteNoiseKernel",
	"StaticFeatureKernel",
	"FeatureKernel",
	# Matern family
	"StaticMatern12Kernel",
	"Matern12Kernel",
	"StaticMatern32Kernel",
	"Matern32Kernel",
	"StaticMatern52Kernel",
	"Matern52Kernel",
	# Operator modules
	"OperatorModule",
	"SumModule",
	"ProductModule",
	# Wrapper modules/kernels
	"WrapperModule",
	"NegModule",
	"ExpModule",
	"LogModule",
	"ActiveDimsModule",
	"ARDKernel",
	"BatchModule",
	"BlockKernel",
	"BlockDiagKernel",
	# Mean functions
	"StaticZeroMean",
	"ZeroMean",
	"StaticConstantMean",
	"ConstantMean",
	"StaticLinearMean",
	"LinearMean",
	"StaticAffineMean",
	"AffineMean",
]