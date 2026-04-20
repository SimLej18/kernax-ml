from __future__ import annotations
import pytest
import jax.numpy as jnp
from kernax.parametrisations import LogExpParametrisation, SoftPlusParametrisation, BoundedParametrisation, ParametrisationChain


@pytest.mark.parametrize("value", [0.1, 1.0, 5.0])
def test_log_exp_roundtrip(value):
	p = LogExpParametrisation()
	assert jnp.allclose(p.unwrap(p.wrap(jnp.array(value))), jnp.array(value), atol=1e-5)


@pytest.mark.parametrize("value", [0.1, 1.0, 5.0])
def test_softplus_roundtrip(value):
	p = SoftPlusParametrisation()
	assert jnp.allclose(p.unwrap(p.wrap(jnp.array(value))), jnp.array(value), atol=1e-5)


@pytest.mark.parametrize("value", [0.2, 0.5, 0.8])
def test_bounded_roundtrip(value):
	p = BoundedParametrisation(lower_bound=jnp.array(0.0), upper_bound=jnp.array(1.0))
	assert jnp.allclose(p.unwrap(p.wrap(jnp.array(value))), jnp.array(value), atol=1e-5)


@pytest.mark.parametrize("value", [.1, 2.5, 3.])
def test_chain_roundtrip(value):
	chain = ParametrisationChain(parametrisations=[BoundedParametrisation(-5, 5), SoftPlusParametrisation()])
	assert jnp.allclose(chain.unwrap(chain.wrap(jnp.array(value))), jnp.array(value), atol=1e-5)
