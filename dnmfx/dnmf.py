import jax
import jax.numpy as jnp
import optax
import zarr
import math
import numpy as np
from jax import jit
from jax import random
import random
from preprocess import Component

def dnmf(global_overlap_lookup_dict, max_iterations, data, batch_size):

    cell_components = []
    trace_components = []
    background_components = []

    for component_index, components in global_overlap_lookup_dict.items():
        # if the connected component has overlaps
        if len(components) != 1:
            local_overlap_lookup_dict = {}
            local_components = [Component((c.begin[0], c.end[0], c.begin[1],
                               c.end[1]), i) for i, c in enumerate(components)]

            for i, c in enumerate(components):
                local_overlap_lookup_dict[i] = local_components

            H, W, B = initialize(local_components, data)

            for i in range(max_iterations):
                component = random.sample(local_components, 1)[0]
                x_batch_dict = create_batch(component, data, batch_size,
                                            local_overlap_lookup_dict, H, W, B)
                #print(f'x_batch_dict - {x_batch_dict}')
                #print(f"x actual: {x_batch_dict['actual'][0].shape}")
                #print(f"x estimate: {x_batch_dict['estimate'][0].shape}")


        '''
        H, W, B = update(x_batch_dict, H, W, B)
        cell_components.append(H)
        trace_components.append(W)
        background_components.append(B)
        '''
    '''
    return { cells: cell_components,
             traces: trace_components,
             backgrounds: background_components}
    '''

def create_batch(component, data, batch_size,
                 local_overlap_lookup_dict, H, W, B):

    num_frames = data.shape[0]
    batched_frames = random.sample(list(range(num_frames)), batch_size)
    x_actual_batch = []
    x_estimate_batch = []
    image_size = int(math.sqrt(data.shape[1]))

    for frame in batched_frames:

        frame_data = data[frame, :].reshape((image_size, image_size))
        x_actual = extract_data(component, frame_data)
        x_actual_batch.append(x_actual)
        x_estimate = estimate_data(component, frame, frame_data,
                                   local_overlap_lookup_dict, H, W, B)
        x_estimate_batch.append(x_estimate)

    return {'actual': x_actual_batch, 'estimate': x_estimate_batch}


def extract_data(component, data):

    height, width, _ = get_component_size(component)
    extracted_data = np.zeros((height, width))
    for i, row in enumerate(range(component.begin[1], component.end[1])):
        for j, col in enumerate(range(component.begin[0], component.end[0])):
            extracted_data[i, j] = data[row, col]

    return extracted_data


def estimate_data(component, frame, data, local_overlap_lookup_dict, H, W, B):

    overlapping_components = local_overlap_lookup_dict[component.index][:]
    # exclude the component itself
    overlapping_components.remove(component)

    x_estimate = B[component.index, :] +\
                 W[frame, component.index] * H[component.index, :]
    height, width, _ = get_component_size(component)
    x_estimate = x_estimate.reshape((height, width))

    for overlapping_component in overlapping_components:
        overlap = get_overlapping_data(component, overlapping_component, data)
        x_estimate += overlap

    return x_estimate


def get_overlapping_data(componentA, componentB, data):
    # extract pixels in region where component B overlaps with compnent A
    height, width, _ = get_component_size(componentA)
    overlap = np.zeros((height, width))

    x_begin_in_range = componentA.begin[0] in range(componentB.begin[0],
                       componentB.end[0])
    x_end_in_range = componentA.end[0] in range(componentB.begin[0],
                     componentB.end[0])
    y_begin_in_range = componentA.begin[1] in range(componentB.begin[1],
                       componentB.end[1])
    y_end_in_range = componentA.end[1] in range(componentB.begin[1],
                     componentB.end[1])

    condition1 = x_begin_in_range and y_end_in_range
    condition2 = x_end_in_range and y_end_in_range
    condition3 = x_begin_in_range and y_begin_in_range
    condition4 = x_end_in_range and y_begin_in_range
    # start and end rows and columns on the image
    if condition1:
        start_row, end_row = componentB.begin[1], componentA.end[1]
        start_col, end_col = componentA.begin[0], componentB.end[0]
    elif condition2:
        start_row, end_row = componentB.begin[1], componentA.end[1]
        start_col, end_col = componentB.begin[0], componentA.end[0]
    elif condition3:
        start_row, end_row = componentA.begin[1], componentB.end[1]
        start_col, end_col = componentA.begin[0], componentB.end[0]
    elif condition4:
        start_row, end_row = componentA.begin[1], componentB.end[1]
        start_col, end_col = componentB.begin[0], componentA.end[0]
    # shift row by y_end; shift column by x_begin
    row_shift, col_shift = componentA.end[1], componentA.begin[0]
    for row in range(start_row, end_row):
        for col in range(start_col, end_col):
            overlap[row-row_shift, col-col_shift] = data[row, col]

    return overlap

def initialize(components, data):

    num_frames = data.shape[0]
    num_components = len(components)
    _, _, component_size = get_component_size(components[0])

    H = np.random.randn(num_components, component_size)
    B = np.random.randn(num_components, component_size)
    W = np.random.randn(num_frames, num_components)

    return H, W, B

def get_component_size(component):

    width = component.end[0]-component.begin[0]
    height = component.end[1]-component.begin[1]

    return width, height, width*height


if __name__ == "__main__":
    # Preprocessing
    bounding_boxes = np.load("bounding_boxes.npy")
    overlap_lookup_dict = create_overlap_lookup_dict(bounding_boxes)

    max_iterations = 1000
    batch_size = 10
    data_path = "ground_truth/ensemble/sequence.zarr"
    data = zarr.load(data_path)
    num_frames, image_size, _ = data.shape
    data = data.reshape(num_frames, image_size**2)

    dnmf(overlap_lookup_dict, max_iterations, data, batch_size)

