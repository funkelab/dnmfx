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
            B_logits=None,
            x_hat_logits=None):

        self.iteration = iteration
        self.loss = loss
        self.grad_H_logits = grad_H_logits
        self.grad_W_logits = grad_W_logits
        self.grad_B_logits = grad_B_logits
        self.H_logits = H_logits
        self.W_logits = W_logits
        self.B_logits = B_logits
        self.x_hat_logits = x_hat_logits

        assert iteration is not None and loss is not None, \
            "Both iteration and loss have to be logged"


class Log():
    """The log of the optimization procedure.

    Contains one instance of :class:`IterationLog` per iteration.
    """

    def __init__(self):

        self.iteration_logs = []

    def add_loss(self,
                 iteration,
                 loss,
                 H_gradient,
                 W_gradient,
                 B_gradient,
                 H_logits,
                 W_logits,
                 B_logits,
                 x_hat_logits):

        self.iteration_logs.append(IterationLog(iteration,
                                                loss,
                                                H_gradient,
                                                W_gradient,
                                                B_gradient,
                                                H_logits,
                                                W_logits,
                                                B_logits,
                                                x_hat_logits))
