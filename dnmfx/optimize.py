import jax
import sys
import jax.numpy as jnp
import optax
import zarr
import math
import numpy as np
from jax import jit
import random
from preprocess import create_components
import funlib.geometry as fg

def dnmf(components, max_iterations, data, batch_size):

    H, W, B = initialize(components, data, 42)
    num_frames = data.shape[0]
    image_size = int(math.sqrt(data.shape[1]))

    for i in range(max_iterations):

        print(f'Num Iteration - {i}')
        component = random.sample(components, 1)[0]
        frame_indices = random.sample(list(range(num_frames)), batch_size)

        for frame_index in frame_indices:

            frame = data[frame_index, :].reshape((image_size, image_size))
            x = extract(component, frame)
            intersect = get_total_intersect(component, frame)
            grad_H, grad_W, grad_B = jax.grad(loss, argnums=(0,1,2))(H, W, B,
                                              x, frame_index, component.index,
                                              intersect)
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


def create_batch(component, frame_indices, image_size, H, W, B):

    x_hat_batch = []
    x_batch = [extract(component, frame_index) for frame_index in frame_indices]

    for frame_index in frame_indices:
        frame = data[frame_index, :].reshape((image_size, image_size))
        total_intersect = get_total_intersect(component, frame)
        x_hat_batch.append(estimate(H, W, B, frame_index, component.index,
                                    total_intersect))

    return x_batch, x_hat_batch


def extract(component, frame):

    extracted = jnp.zeros(component.bounding_box.shape)
    start_col, start_row = component.bounding_box.get_begin()
    end_col, end_row = component.bounding_box.get_end()

    for i, row in enumerate(range(start_row, end_row)):
        for j, col in enumerate(range(start_col, end_col)):
            extracted.at[i, j].set(frame[row, col])

    return extracted.flatten()


def estimate(H, W, B, frame_index, component_index, intersect):

    estimated = B[component_index] +\
                W[component_index][frame_index] * H[component_index]

    return estimated + intersect


def get_total_intersect(component, frame):

    total_intersect = jnp.zeros((component.bounding_box.shape))
    for overlapping_component in component.overlapping_components:
        total_intersect.at[:, :].add(get_intersect(component.bounding_box,
                                overlapping_component.bounding_box,
                                frame))

    return total_intersect.flatten()


def loss(H, W, B, x, frame_index, component_index, intersect):

    x_hat = estimate(H, W, B, frame_index, component_index, intersect)

    return jnp.linalg.norm(x - x_hat).sum()


def update(H, W, B, grad_H, grad_W, grad_B, component_index):

    H[component_index].at[:].add(-grad_H[component_index])
    W[component_index].at[:].add(-grad_W[component_index])
    B[component_index].at[:].add(-grad_B[component_index])

    return H, W, B


def get_intersect(componentA, componentB, frame):

    intersect = jnp.zeros(componentA.shape)
    intersect_roi = componentA.intersect(componentB)
    row_shift, col_shift = componentA.get_end()[1], componentA.get_begin()[0]

    start_col, start_row = intersect_roi.get_begin()
    end_col, end_row = intersect_roi.get_end()

    for row in range(start_row, end_row):
        for col in range(start_col, end_col):
            intersect.at[row-row_shift, col-col_shift].set(frame[row, col])

    return intersect



if __name__ == "__main__":

    bounding_boxes = np.load("bounding_boxes.npy")
    bounding_boxes[2, 1] = 516
    bounding_boxes[9, 1] = 516
    bounding_boxes = [fg.Roi((x_b, y_b), (x_e-x_b, y_e-y_b))
                      for x_b, x_e, y_b, y_e in bounding_boxes]

    components = create_components(bounding_boxes)
    max_iterations = 10
    batch_size = 10
    data_path = "ground_truth/ensemble/sequence.zarr"
    data = jnp.asarray(zarr.load(data_path))
    num_frames, image_size, _ = data.shape
    data = data.reshape(num_frames, image_size**2)
    dnmf(components, max_iterations, data, batch_size)

