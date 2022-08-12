from datetime import datetime
import jax.numpy as jnp
from .optimize import dnmf
from .parameters import Parameters
from .groups import get_groups
from .io import read_dataset
from .initialize import initialize_normal


def fit(
        dataset_path,
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

        The optimization result of the dataset (i.e. H, W, B) and the losses stored as
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

    if parameters.random_seed is None:
        parameters.random_seed = int(datetime.now().strftime("%Y%m%d%H%M%S"))
    else:
        parameters.random_seed = random_seed

    H_groups = []
    W_groups = []
    B_groups = []
    log_groups = []

    groups = get_groups(dataset_path)
    component_group_index_pairings = {}
    for group_index, group in enumerate(groups):
        for component in group:
            component_group_index_pairings[component.index] = group_index

    for group in groups:
        H_group, W_group, B_group, log_group = fit_group(
                                                    group,
                                                    dataset_path,
                                                    parameters)
        H_groups.append(H_group)
        W_groups.append(W_group)
        B_groups.append(B_group)
        log_groups.append(log_group)

    H, W, B = assemble(component_group_index_pairings,
                       H_groups, B_groups, W_groups)

    return H, W, B, log_groups


def fit_group(component_descriptions,
              dataset_path,
              parameters):
    """Use NMF to estimate the components and traces for a group from the dataset
    given as a list of :class: `ComponentDescription`

    Args:

        component_descriptions (list of :class: `ComponentDescription`):
            A list of :class: `ComponentDescription` that form a group in the
            dataset.

        dataset_path (string):
            The path to the zarr container containing the dataset. Should have
            a `sequence` dataset of shape `(t, [[z,], y,] x)` and a
            `component_locations` dataset of shape `(n, 2, d)`, where `n` is
            the number of components and `d` the number of spatial dimensions.
            `component_locations` stores the begin and end of each component,
            i.e., `component_locations[1, 0, :]` is the begin of component `1`
            and `component_locations[1, 1, :]` is its end.

        parameters (:class: `Parameters`):
            Parameters to control the optimization.

    Returns:

        The optimization result of a group from the dataset (i.e. `H_group`, `W_group`,
        `B_group`) and the losses stored as :class: `Log`.
    """

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
            dataset,
            component_descriptions,
            parameters,
            H_logits,
            W_logits,
            B_logits)

    return H, W, B, log


def assemble(component_group_index_pairings,
             H_groups,
             B_groups,
             W_groups):
    """Assemble optimization results from all groups in the dataset.

    Args:

        component_group_index_pairings (dictionary):
            A dictionary of size number of the components that map component index
            (key) to group index (value).

        H_groups (list):
            A list of `H_group` obtained from optimization result of a single group
            in the dataset.

        B_groups (list):
            A list of `B_group` obtained from optimization result of a single group
            in the dataset.

        W_groups (list):
            A list of `W_group` obtained from optimization result of a single group
            in the dataset.

    Returns:
         The optimization result of the dataset (i.e. H, W, B).
    """

    num_components = len(component_group_index_pairings)
    group_index = component_group_index_pairings[0]

    H = H_groups[group_index][0]
    B = B_groups[group_index][0]
    W = W_groups[group_index][:, 0].reshape(-1, 1)

    for component_index in range(1, num_components):
        H = jnp.vstack((H_groups[group_index][component_index], H))
        B = jnp.vstack((B_groups[group_index][component_index], B))
        W = jnp.hstack((W, W_groups[group_index][:, component_index].reshape(-1, 1)))

    return H, W, B
