class IterationLog():
    """The loss (and possibly other stats in the future) for a single
    iteration.
    """

    def __init__(self, iteration, loss):
        self.iteration = iteration
        self.loss = loss


class Log():
    """The log of the optimization procedure.

    Contains one instance of :class:`IterationLog` per iteration.
    """

    def __init__(self):

        self.iteration_logs = []

    def add_loss(self, iteration, loss):

        self.iteration_logs.append(IterationLog(iteration, loss))
