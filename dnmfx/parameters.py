import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
class Parameters():

    def __init__(
            self,
            max_iteration=1000,
            min_loss=1e-4,
            batch_size=10,
            step_size=1e-3,
            l1_weight=0.1):

        self.max_iteration = max_iteration
        self.min_loss = min_loss
        self.batch_size = batch_size
        self.step_size = step_size
        self.l1_weight = l1_weight

    def tree_flatten(self):
        children = (self.max_iteration,
                    self.min_loss,
                    self.batch_size,
                    self.step_size,
                    self.l1_weight)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

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

    def __init__(self, iteration_loss):
        if aggregate_loss == None:
            aggregate_loss = iteration_loss
        else:
            aggregate_loss = np.stack((aggregate_loss, iteration_loss))
