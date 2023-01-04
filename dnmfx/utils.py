import jax.numpy as jnp


def log1pexp(x):
    return jnp.log1p(jnp.exp(x))


def sigmoid(Z):
    A = 1 / (1 + (jnp.exp((-Z))))
    return A
