import numpy as np
import zarr
#import shutil
from scipy.optimize import linear_sum_assignment


def evaluate(reconstruction_path, ground_truth_path, num_components,
             num_frames, show_component_errors=True):

    component_matrix = np.zeros((num_components, num_components))
    trace_matrix = np.zeros((num_components, num_components))
    '''
    # copy background to the `components` directory
    shutil.copytree(f"{ground_truth_path}/background.zarr",
            f"{ground_truth_path}/components/component{num_components-1}.zarr")
    '''
    # find the L2 norm of reconstruction vs. ground truth
    for i in range(num_components):
        if i == num_components - 1:
            ground_truth_trace = np.zeros(num_frames)
            background_path = f"{ground_truth_path}/background.zarr"
            ground_truth_component = zarr.load(background_path).flatten()
        else:
            ground_truth_trace_path = f"{ground_truth_path}/traces/trace{i}.zarr"
            ground_truth_trace = zarr.load(ground_truth_trace_path)
            ground_truth_component_path = \
                f"{ground_truth_path}/components/component{i}.zarr"
            ground_truth_component = \
                    zarr.load(ground_truth_component_path).flatten()

        for j in range(num_components):
            reconstructed_trace_path = \
            f"{reconstruction_path}/traces/trace{j}.zarr"
            reconstructed_trace = zarr.load(reconstructed_trace_path)
            reconstructed_component_path = \
            f"{reconstruction_path}/components/component{j}.zarr"
            reconstructed_component = \
            zarr.load(reconstructed_component_path).flatten()

            component_matrix[i][j] = np.linalg.norm(reconstructed_component -
                                               ground_truth_component)
            trace_matrix[i][j] = np.linalg.norm(reconstructed_trace -
                                                ground_truth_trace)
    # find optimal assigment
    row_indices, col_indices = linear_sum_assignment(component_matrix)
    # matched pairs of reconstruction and ground truth components
    matched_pairs = list(zip(row_indices, col_indices))

    total_component_reconstruction_error = component_matrix[row_indices, col_indices].sum()
    total_trace_reconstruction_error = trace_matrix[row_indices, col_indices].sum()

    results = {
                "matched_pairs": matched_pairs,
                "total_component_reconstruction_error": total_component_reconstruction_error,
                "total_trace_reconstruction_error": total_trace_reconstruction_error
              }

    if show_component_errors:
        component_reconstruction_errors = get_component_reconstruction_error(
                                                matched_pairs,
                                                reconstruction_path,
                                                ground_truth_path
                                            )
        results["component_errors"] = component_reconstruction_errors

    return results

def get_component_reconstruction_error(matched_pairs, reconstruction_path,
                                       ground_truth_path):
    component_reconstruction_errors = {}
    for matched_pair in matched_pairs:
        id_ground_truth, id_reconstruction = matched_pair
        ground_truth_component = \
        zarr.load(f"{ground_truth_path}/components/component{id_ground_truth}.zarr")
        reconstructed_component = \
        zarr.load(f"{reconstruction_path}/components/component{id_reconstruction}.zarr")
        reconstruction_error = np.linalg.norm(reconstructed_component -
                                              ground_truth_component)
        component_reconstruction_errors[id_ground_truth] = reconstruction_error

    return component_reconstruction_errors
