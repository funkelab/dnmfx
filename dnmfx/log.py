class IterationLog():
    """The loss (and possibly other stats in the future) for a single
    iteration.
    """

    def __init__(self,
            iteration,
            loss,
            grad_H_logits=None,
            grad_W_logits=None,
            grad_B_logits=None,
            H_logits=None,
            W_logits=None,
            B_logits=None):

        self.iteration = iteration
        self.loss = loss
        self.grad_H_logits = grad_H_logits
        self.grad_W_logits = grad_W_logits
        self.grad_B_logits = grad_B_logits
        self.H_logits = H_logits
        self.W_logits = W_logits
        self.B_logits = B_logits

        assert iteration is not None and loss is not None, \
            "Both iteration and loss have to be logged"


class Log():
    """The log of the optimization procedure.

    Contains one instance of :class:`IterationLog` per iteration.
    """

    def __init__(self):

        self.iteration_logs = []

    def log_iteration(self,
                      iteration,
                      loss,
                      log_gradients=False,
                      H_gradient=None,
                      W_gradient=None,
                      B_gradient=None,
                      H_logits=None,
                      W_logits=None
                      B_logits=None):

        if log_gradients:

            assert None not in [H_gradient, W_gradient, B_gradient], \
                    "Make sure gradients are not None"

            self.iteration_logs.append(IterationLog(iteration,
                                                    loss,
                                                    H_gradient,
                                                    W_gradient,
                                                    B_gradient,
                                                    H_logits,
                                                    W_logits,
                                                    B_logits))
        else: self.iteration_logs.append(IterationLog(iteration, loss))
