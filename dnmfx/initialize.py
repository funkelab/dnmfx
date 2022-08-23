from .utils import log1pexp
import jax


def initialize_normal(num_components, num_frames, component_size, random_seed):
    """Initialize factors H, B, W with random numbers from a normal distribution to
    be optimized with distributed NMF, such that

                    X[t,s] ≈ W[t,n]@H[n,s] + 1[t,n]@B[n,s]
                           = X_hat[t,s].

    Args:
        num_components (int):
            Number of components in the dataset.

        num_frames (int):
            Number of time frames in the dataset.

        component_size (int):
            Size of the component.

        random_seed (int or None):
            Random seed for generating a split key used for sampling; if set to
            `None`, a random integer will be used as replacement.

    Returns:

        Initial guesses of `H_logits`, `W_logits`, `B_logits`.
    """

    key = jax.random.PRNGKey(random_seed)

    key, subkey = jax.random.split(key)
    H_logits = log1pexp(jax.random.normal(
        subkey,
        shape=(num_components, component_size)))

    key, subkey = jax.random.split(key)
    B_logits = log1pexp(jax.random.normal(
        subkey,
        shape=(num_components, component_size)))

    _, subkey = jax.random.split(key)
    W_logits = log1pexp(jax.random.normal(
        subkey,
        shape=(num_frames, num_components))).T

    return H_logits, W_logits, B_logits
