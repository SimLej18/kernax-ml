class StaticZeroMean(StaticAbstractMean):
	@classmethod
	@filter_jit
	def scalar_mean(cls, mean: AbstractMean, x: Array) -> Array:
		return jnp.array(0.)

class ZeroMean(AbstractMean):
	static_class = StaticZeroMean


class StaticConstantMean(StaticAbstractMean):
	@classmethod
	@filter_jit
	def scalar_mean(cls, mean: AbstractMean, x: Array) -> Array:
		return mean.constant

class ConstantMean(AbstractMean):
	constant: Array = eqx.field(converter=jnp.asarray)
	static_class = StaticConstantMean

	def __init__(self, constant=0., **kwargs):
		super().__init__(**kwargs)
		self.constant = jnp.asarray(constant)



class StaticLinearMean(StaticAbstractMean):
	@classmethod
	@filter_jit
	def scalar_mean(cls, mean: AbstractMean, x: Array) -> Array:
		return mean.slope * x

class LinearMean(AbstractMean):
	slope: Array = eqx.field(converter=jnp.asarray)
	static_class = StaticLinearMean

	def __init__(self, slope=0., **kwargs):
		super().__init__(**kwargs)
		self.slope = jnp.asarray(slope)

class StaticAffineMean(StaticAbstractMean):
	@classmethod
	@filter_jit
	def scalar_mean(cls, mean: AbstractMean, x: Array) -> Array:
		return mean.slope * x + mean.intercept

class AffineMean(AbstractMean):
	slope: Array = eqx.field(converter=jnp.asarray)
	intercept: Array = eqx.field(converter=jnp.asarray)
	static_class = StaticAffineMean

	def __init__(self, slope=0., intercept=0., **kwargs):
		super().__init__(**kwargs)
		self.slope = jnp.asarray(slope)
		self.intercept = jnp.asarray(intercept)
