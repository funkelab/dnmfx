from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class Parameters():

    def __init__(
            self,
            max_iteration,
            min_loss,
            batch_frames,
            batch_components,
            step_size,
            l1_weight,
            log_every,
            log_gradients,
            random_seed):

        self.max_iteration = max_iteration
        self.min_loss = min_loss
        self.batch_frames = batch_frames
        self.batch_components = batch_components
        self.step_size = step_size
        self.l1_weight = l1_weight
        self.log_every = log_every
        self.log_gradients = log_gradients
        self.random_seed = random_seed

    def tree_flatten(self):
        children = (self.max_iteration,
                    self.min_loss,
                    self.batch_frames,
                    self.batch_components,
                    self.step_size,
                    self.l1_weight,
                    self.log_every,
                    self.log_gradients,
                    self.random_seed)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
