class IterationLog():
    """The loss (and possibly other stats in the future) for a single
    iteration.
    """

    def __init__(self,
                 iteration,
                 reconstruction_loss,
                 component_loss,
                 trace_loss,
                 grad_H_logits,
                 grad_W_logits,
                 grad_B_logits,
                 H_logits,
                 W_logits,
                 B_logits):

        self.iteration = iteration
        self.reconstruction_loss = reconstruction_loss
        self.component_loss = component_loss
        self.trace_loss = trace_loss
        self.grad_H_logits = grad_H_logits
        self.grad_W_logits = grad_W_logits
        self.grad_B_logits = grad_B_logits
        self.H_logits = H_logits
        self.W_logits = W_logits
        self.B_logits = B_logits

        assert iteration is not None and reconstruction_loss is not None, \
            "Both iteration and loss have to be logged"


class Log():
    """The log of the optimization procedure.

    Contains one instance of :class:`IterationLog` per iteration.
    """

    def __init__(self):

        self.iteration_logs = []

    def log_iteration(self,
                      iteration,
                      reconstruction_loss,
                      component_loss=None,
                      trace_loss=None,
                      H_gradient=None,
                      W_gradient=None,
                      B_gradient=None,
                      H_logits=None,
                      W_logits=None,
                      B_logits=None):

        self.iteration_logs.append(
                IterationLog(
                             iteration,
                             reconstruction_loss,
                             component_loss,
                             trace_loss,
                             H_gradient,
                             W_gradient,
                             B_gradient,
                             H_logits,
                             W_logits,
                             B_logits))
