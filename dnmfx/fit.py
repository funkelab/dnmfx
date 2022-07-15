import numpy as np
from nmfx import parameters, nmf
from .io import read_dataset


def fit(
        data_path,
        max_iteration,
        min_loss,
        batch_size,
        step_size,
        l1_weight,
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

        log_every (int):
            How often to print a log statement during optimization.

    Returns:

        A dictionary with "H" (the estimated components), "W" (the estimated
        traces), and "log" (statistics per iteration).
    """

    nmf_parameters = parameters.Parameters()
    nmf_parameters.max_iter = max_iteration
    nmf_parameters.min_loss = min_loss
    nmf_parameters.batch_size = batch_size
    nmf_parameters.step_size = step_size
    nmf_parameters.l1_W = l1_weight

    dataset = read_dataset(data_path)

    # Initialize H, W
    H = np.random.randn(dataset.num_frames, dataset.num_components)
    W = np.random.randn(dataset.num_components, dataset.frame_roi.get_size())

    initial_values = {"H": H, "W": W}
    H, W, log = nmf(
        dataset.sequence,
        dataset.num_components,
        nmf_parameters,
        log_every,
        initial_values)

    return {"H": H, "W": W, "log": log}
