import jax
import jax.numpy as jnp
import zarr
import math
import numpy as np
from jax import jit
import random
from dnmfx.preprocess import create_components
import funlib.geometry as fg


def dnmf(components, max_iterations, data, batch_size, random_seed):

    H, W, B = initialize(components, data, random_seed)
    num_frames = data.shape[0]
    image_size = int(math.sqrt(data.shape[1]))

    for i in range(max_iterations):
        print(f'Num Iteration - {i}')
        component = random.sample(components, 1)[0]
        frame_indices = random.sample(list(range(num_frames)), batch_size)
        grad_H, grad_W, grad_B = jax.grad(loss, argnums=(0,1,2))(H, W, B, data,
                                          frame_indices, component, image_size)
        H, W, B = update(H, W, B, grad_H, grad_W, grad_B, component.index)

    return H, W, B


def initialize(components, data, random_seed):

    num_frames = data.shape[0]
    num_components = len(components)
    component_size = components[0].bounding_box.get_size()

    key = jax.random.PRNGKey(random_seed)

    H = [jax.random.normal(key, (component_size,)) for i in range(num_components)]
    W = [jax.random.normal(key, (num_frames,)) for i in range(num_components)]
    B = [jax.random.normal(key, (component_size,)) for i in range(num_components)]

    return H, W, B


def update(H, W, B, grad_H, grad_W, grad_B, component_index):

    H[component_index].at[:].add(-grad_H[component_index])
    W[component_index].at[:].add(-grad_W[component_index])
    B[component_index].at[:].add(-grad_B[component_index])

    return H, W, B


def loss(H, W, B, data, frame_indices, component, image_size):

    x_batch = [extract(component, frame_index, data, image_size)
               for frame_index in frame_indices]
    x_hat_batch = []

    for frame_index in frame_indices:
        frame = data[frame_index, :].reshape((image_size, image_size))
        x_hat_batch.append(estimate(H, W, B, frame,
                                    frame_index, component))

    return jnp.linalg.norm(jnp.asarray(x_batch) -
                           jnp.asarray(x_hat_batch)).sum()


def extract(component, frame_index, data, image_size):

    frame = data[frame_index, :].reshape((image_size, image_size))
    extracted = jnp.zeros(component.bounding_box.shape)
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

    total_intersect = jnp.zeros((component.bounding_box.shape))
    for overlapping_component in component.overlapping_components:
        total_intersect.at[:, :].add(get_intersect(component.bounding_box,
                                     overlapping_component.bounding_box,
                                     frame))

    return total_intersect.flatten()


def get_intersect(bounding_box_A, bounding_box_B, frame):

    intersect = jnp.zeros(bounding_box_A.shape)
    intersect_roi = bounding_box_A.intersect(bounding_box_B)

    row_shift = bounding_box_A.get_end()[1]
    col_shift = bounding_box_A.get_begin()[0]

    start_col, start_row = intersect_roi.get_begin()
    end_col, end_row = intersect_roi.get_end()

    for row in range(start_row, end_row):
        for col in range(start_col, end_col):
            intersect.at[row-row_shift, col-col_shift].set(frame[row, col])

    return intersect
