from dnmfx.io import read_dataset
import numpy as np
from scipy.optimize import linear_sum_assignment


def evaluate(H, B, W, dataset):

    """
    Get the number of component and component background ID mismatches and
    find the reconstruction error per component and per trace.

    Args:

        H (array-like, shape `(k, w*h)`):

            The factorized matrix that contains all decomposed components.

        B (array-like, shape `(k, w*h)`):

            The factorized matrix that contains the backgrounds of all decomposed
            components.

        W (array-like, shape `(t, k)`):

            The factorized matrix that contains the traces of all decomposed
            components.

        dataset (:class: `Dataset`):

            Dataset to be factorized.
    """

    components = dataset.components
    background = dataset.background
    traces = dataset.traces
    bounding_boxes = dataset.bounding_boxes
    k = dataset.num_components

    H = H.reshape(-1, *components[0].shape)
    B = B.reshape(-1, *components[0].shape)

    component_loss = {i: np.linalg.norm(components[i, :] - H[i, :])
                      for i in range(k)}
    trace_loss = {i: np.linalg.norm(traces[i, :] - W[:, i])
                      for i in range(k)}

    return component_loss, trace_loss
