import numpy as np
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class IterationLog():

    """
    Construct the iteration loss matrix with shape (k, T), where k is the number of
    connected components, T the batch size, whose each element (e.g. x) is the
    reconstruction loss of a sampled component in a given connected component at a
    time frame.

            t_0  t_1  .    .   .   t_T
    cc_0 |   x                           |
    cc_1 |                               |
    .    |                               |
    .    |                               |
    .    |                               |
    cc_k |                               |
    """

    def __init__(self, num_connected_components, batch_size):
        self.iteration_loss = np.zeros((num_connected_components,
                                        batch_size))

    def tree_flatten(self):
        children = (self.iteration_loss)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

class AggregateLog():

    def __init__(self):
        aggregate_loss = np.array([])
