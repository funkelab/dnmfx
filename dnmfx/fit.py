from .component_description import create_component_description
from .optimize import dnmf
from .parameters import Parameters
from .io import read_dataset


def fit(
        data_path,
        max_iteration,
        min_loss,
        batch_size,
        step_size,
        l1_weight,
        random_seed=None,
        log_every=1):
    """Use distributed NMF to estimate the components and traces for a dataset
    given as a zarr container.

    Args:

        data_path (string):
            The path to the zarr container containing the dataset. Should have
            a `sequence` dataset of shape `(t, [[z,], y,] x)` and a
            `component_locations` dataset of shape `(n, 2, d)`, where `n` is
            the number of components and `d` the number of spatial dimensions.
            `component_locations` stores the begin and end of each component,
            i.e., `component_locations[1, 0, :]` is the begin of component `1`
            and `component_locations[1, 1, :]` is its end.

        max_iteration (int):
            The maximum number of iterations to optimize for.

        min_loss (float):
            The loss value at which to stop the optimization.

        batch_size (int):
            The number of frames to consider at once for each component during
            the stochastic gradient estimation.

        step_size (float):
            The size of the gradient updates.

        l1_weight (float):
            The influence of the L1 regularizer on the components and traces.

        random_seed (int):
            A random seed for the initialization of `H` and `W`. If not given,
            a different random seed will be used each time.

        log_every (int):
            How often to print a log statement during optimization.

    Returns:

        A dictionary with "H" (the estimated components), "W" (the estimated
        traces), and "log" (statistics per iteration).
    """

    parameters = Parameters()
    parameters.max_iteration = max_iteration
    parameters.min_loss = min_loss
    parameters.batch_size = batch_size
    parameters.step_size = step_size
    parameters.l1_weight = l1_weight

    dataset = read_dataset(data_path)
    component_description = \
        create_component_description(dataset.bounding_boxes)

    H, W, B, log = dnmf(
        dataset.sequence,
        component_description,
        parameters,
        log_every=10,
        random_seed=random_seed)

    return {"H": H, "W": W, "B": B, "log": log}
