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
