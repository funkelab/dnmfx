import jax.numpy as jnp

log1pexp = lambda x: jnp.log1p(jnp.exp(x))
def sigmoid(Z):
    A=1/(1+(jnp.exp((-Z))))
    return A
