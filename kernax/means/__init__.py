from .AffineMean import AffineMean, StaticAffineMean
from .ConstantMean import ConstantMean, StaticConstantMean
from .LinearMean import LinearMean, StaticLinearMean
from .ZeroMean import StaticZeroMean, ZeroMean

__all__ = [
	"StaticZeroMean",
	"ZeroMean",
	"StaticConstantMean",
	"ConstantMean",
	"StaticLinearMean",
	"LinearMean",
	"StaticAffineMean",
	"AffineMean",
]