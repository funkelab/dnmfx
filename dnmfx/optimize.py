from datetime import datetime
import jax
import jax.numpy as jnp
import math
import random
from .log import IterationLog, AggregateLog

def dnmf(sequence,
        component_info,
        parameters,
        num_cells,
        num_frames,
        component_dims,
        log_every=10,
        random_seed=None):

    if random_seed is None:
        random_seed = datetime.now().toordinal()

    H, W, B = initialize(num_cells, num_frames, component_dims, random_seed)
    connected_components = get_connected_components(component_info)
    aggregate_log = AggregateLog()

    for i in range(parameters.max_iteration):

        print(f'Iteration {i}')
        iteration_log = IterationLog(num_cells, parameters.batch_size)

        for connnect_component_index, connected_component in \
                                                enumerate(connected_components):
            component = random.sample(connected_component, 1)[0]
            frame_indices = random.sample(list(range(num_frames)),
                                          parameters.batch_size)
            iteration_log, grad_H, grad_W, grad_B = jax.grad(
                                        loss,
                                        argnums=(0,1,2),
                                        has_aux=True)(
                                        H, W, B,
                                        sequence,
                                        frame_indices,
                                        component,
                                        connected_component_index,
                                        iteration_log)
            H, W, B = update(H, W, B, grad_H, grad_W, grad_B, component.index)

        if i % 10 == 0:
            aggregate_log.aggregate_loss.stack(aggregate_loss,
                                           iteration_log.iteration_loss)

    return H, W, B, aggregate_log


def get_connected_components(component_info):
    sorted_component_info = sorted(component_info, key=lambda x:
            len(x.overlapping_components))
    return run_depth_first_search(sorted_component_info,
                                  sorted_component_info,
                                  sorted_component_info[0],
                                  [],
                                  [])

def run_depth_first_search(unvisited,
                           component_info,
                           component,
                           connected_component,
                           connected_components):
    index = component.index
    if component in unvisited:
        unvisited.remove(component)
        overlaps = component.overlapping_components

        if len(overlaps) == 0:
            connected_components.append([component])
        else:
            all_overlaps_added = True
            for overlap in overlaps:
                if overlap not in connected_component:
                    all_overlaps_added = False
                    connected_component.append(overlap)
                    run_depth_first_search(unvisited,
                                           component_info,
                                           overlap,
                                           connected_component,
                                           connected_components)
            if all_overlaps_added:
                connected_components.append(connected_component)
                if len(unvisited) != 0:
                    run_depth_first_search(unvisited,
                                           component_info,
                                           unvisited[0],
                                           [],
                                           connected_components)
    elif len(unvisited) != 0:
        connected_components.append(connected_component)
        run_depth_first_search(unvisited,
                               component_info,
                               unvisited[0],
                               [],
                               connected_components)

    return connected_components


def initialize(num_cells, num_frames, component_dims, random_seed):

    key = jax.random.PRNGKey(random_seed)

    H = [jax.random.normal(key, component_dims) for i in range(num_cells)]
    W = [jax.random.normal(key, (num_frames,)) for i in range(num_cells)]
    B = [jax.random.normal(key, component_dims) for i in range(num_cells)]

    return H, W, B


def update(H, W, B, grad_H, grad_W, grad_B, component_index):

    H[component_index].at[:].add(-grad_H[component_index])
    W[component_index].at[:].add(-grad_W[component_index])
    B[component_index].at[:].add(-grad_B[component_index])

    return H, W, B


def loss(
        H, W, B,
        sequence,
        frame_indices,
        component,
        connected_component_index,
        iteration_log):

    for t, frame_index in enumerate(frame_indices):
        frame = sequence[frame_index, :, :]
        x = extract(component, frame)
        x_hat = estimate(H, W, B, frame, frame_index, component))
        diff = jnp.linalg.norm(x-x_hat)
        iteration_log.iteration_loss[connected_component_index][t] = diff

    return (iteration_log.iteration_loss[connected_component_index, :].sum(),
            iteration_log)

def extract(component, frame):

    extracted = jnp.zeros(component.bounding_box.shape, dtype=jnp.float32)
    start_col, start_row = component.bounding_box.get_begin()
    end_col, end_row = component.bounding_box.get_end()

    for i, row in enumerate(range(start_row, end_row)):
        for j, col in enumerate(range(start_col, end_col)):
            extracted.at[i, j].set(frame[row, col])

    return extracted.flatten()


def estimate(H, W, B, frame, frame_index, component):

    return B[component.index] + \
           W[component.index][frame_index] * H[component.index] + \
           get_total_intersect(component, frame)


def get_total_intersect(component, frame):

    total_intersect = jnp.zeros(component.bounding_box.shape, dtype=jnp.float32)
    for overlapping_component in component.overlapping_components:
        total_intersect.at[:, :].add(get_intersect(component.bounding_box,
                                     overlapping_component.bounding_box,
                                     frame))

    return total_intersect.flatten()


def get_intersect(bounding_box_A, bounding_box_B, frame):

    intersect = jnp.zeros(bounding_box_A.shape, dtype=jnp.float32)
    intersect_roi = bounding_box_A.intersect(bounding_box_B)

    row_shift = bounding_box_A.get_end()[1]
    col_shift = bounding_box_A.get_begin()[0]

    start_col, start_row = intersect_roi.get_begin()
    end_col, end_row = intersect_roi.get_end()

    for row in range(start_row, end_row):
        for col in range(start_col, end_col):
            intersect.at[row-row_shift, col-col_shift].set(frame[row, col])

    return intersect
