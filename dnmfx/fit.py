from datetime import datetime
import pickle
from .optimize import dnmf
from .parameters import Parameters
from .io import read_dataset
from .initialize import initialize_normal


def fit(
        dataset_path,
        groups_path,
        group_index,
        max_iteration=10000,
        min_loss=1e-4,
        batch_size=10,
        step_size=1e-1,
        l1_weight=0,
        log_every=100,
        log_gradients=False,
        random_seed=None):
    """Use distributed NMF to estimate the components and traces for a dataset
    given as a zarr container.

    Args:

        dataset_path (string):
            The path to the zarr container containing the dataset. Should have
            a `sequence` dataset of shape `(t, [[z,], y,] x)` and a
            `component_locations` dataset of shape `(n, 2, d)`, where `n` is
            the number of components and `d` the number of spatial dimensions.
            `component_locations` stores the begin and end of each component,
            i.e., `component_locations[1, 0, :]` is the begin of component `1`
            and `component_locations[1, 1, :]` is its end.

        groups_path (string):
            The path to the pickle file containing the groups of the dataset. Should
            be a list of lists; the list length is the number of groups; each list
            contains an uncertain number of objects :class: `ComponentDescription`
            that are members within a group.

        group_index (int):
            The index indicating which group is to be fitted.

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

            How often to print iteration statistics.

        log_gradients (bool):

            Whether to record gradients and factor matrices (i.e. H, B, W) after the
            1st iteration.

        random_seed (int):

            A random seed for the initialization of `H` and `W`. If not given,
            a different random seed will be used each time.

    Returns:

        The maxtrix factors of the dataset (i.e. H, W, B) and the losses stored as
        :class: `Log`.
    """

    parameters = Parameters()
    parameters.max_iteration = max_iteration
    parameters.min_loss = min_loss
    parameters.batch_size = batch_size
    parameters.step_size = step_size
    parameters.l1_weight = l1_weight
    parameters.log_every = log_every
    parameters.log_gradients = log_gradients
    parameters.random_seed = random_seed

    if parameters.random_seed is None:
        parameters.random_seed = int(datetime.now().strftime("%Y%m%d%H%M%S"))
    else:
        parameters.random_seed = random_seed

    with open(groups_path, "rb") as f:
        content = f.read()
    groups = pickle.loads(content)

    component_descriptions = groups[group_index]

    component_size = None
    for description in component_descriptions:
        size = description.bounding_box.get_size()
        if component_size is not None:
            assert component_size == size, \
                    "Only components of the same size are supported for now"
        else:
            component_size = size

    dataset = read_dataset(dataset_path)
    sequence = dataset.sequence
    num_frames = sequence.shape[0]
    num_components = dataset.num_components

    H_logits, W_logits, B_logits = initialize_normal(
            num_components,
            num_frames,
            component_size,
            parameters.random_seed)

    H, W, B, log = dnmf(
            sequence,
            component_descriptions,
            parameters,
            H_logits,
            W_logits,
            B_logits)

    return H, W, B, log
