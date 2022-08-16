from dnmfx.io import read_dataset
import numpy as np
from scipy.optimize import linear_sum_assignment


def evaluate(H, W, dataset):
    """Get the number of component and component background ID mismatches and
    find the reconstruction error per component and per trace.

    Args:

        H (array-like, shape `(k, w*h)`):
            The factorized matrix that contains all decomposed components.

        W (array-like, shape `(t, k)`):
            The factorized matrix that contains the traces of all decomposed
            components.

        dataset (:class: `Dataset`):
            Dataset to be factorized.

    Returns:

        component_loss (dictionary):
            A dictionary that corresponds each component index (key) to its loss
            (value) that is the L2 distance between the ground truth (if known) and
            component estimation from optmization.

        trace_loss (dictionary):
            A dictionary that corresponds each component index (key) to its loss
            (value) that is the L2 distance between the ground truth (if known) and
            trace estimation from optmization.
    """

    components = dataset.components
    traces = dataset.traces
    k = dataset.num_components
    num_pixels = components.size
    H = H.reshape(-1, *components[0].shape)

    component_loss = {i: float(np.linalg.norm(components[i, :] - H[i, :])/num_pixels)
                      for i in range(k)}
    trace_loss = {i: float(np.linalg.norm(traces[i, :] - W[:, i])/num_pixels)
                      for i in range(k)}

    return component_loss, trace_loss
