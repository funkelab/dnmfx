import numpy as np
import os
import cv2
from scipy.optimize import linear_sum_assignment


def evaluate(reconstruction_path, ground_truth_path, num_components):

    cell_matrix = np.zeros((num_components, num_components))
    trace_matrx = np.zeros((num_components, num_components))

    ground_truth_traces = np.load(f"{ground_truth_path}/traces/traces.npy")
    num_frames = ground_truth_traces.shape[1]
    reconstructed_traces = np.load(f"{reconstruction_path}/traces.npy").reshape(num_components,num_frames)

    # make up background trace to match dimensionality
    ground_truth_traces = np.concatenate((ground_truth_traces,
                                    np.zeros((1, num_frames))), axis=0)

    # find the L2 norm of reconstruction vs. ground truth
    for i in range(num_components):        
        reconstructed_cell = cv2.imread(f"{reconstruction_path}/cell{i}.jpg").flatten()
        reconstructed_trace = reconstructed_traces[i, :]

        for j in range(num_components):
            ground_truth_cell = cv2.imread(f"{ground_truth_path}/singles/cell{j}.jpg").flatten()
            ground_truth_trace = traces_grdtru[j, :]

            cell_matrix[i][j] = np.linalg.norm(reconstructed_cell - ground_truth_cell)
            trace_matrix[i][j] = np.linalg.norm(reconstructed_trace - ground_truth_cell)

    # find optimal assigment
    row_indices, col_indices = linear_sum_assignment(cell_matrix)
    
    # matched pairs of reconstruction and ground truth components
    matched_pairs = list(zip(row_indices, col_indices))
    
    cell_reconstruction_error = cell_matrix[row_indices, col_indices].sum()
    trace_reconstruction_error = trace_mtx[row_indices, col_indices].sum()

    results = {
                "matched_pairs": matched_pairs,
                "cell_reconstruction_error": cell_reconstruction_error,
                "trace_reconstruction_error": trace_reconstruction_error
              }

    return results


if __name__ == "__main__":

    reconstruction_path = "reconstruction"
    ground_truth_path = "ground_truth"

    num_components = 11

    results = evaluate(reconstruction_path, groud_truth_path, num_components)
    print(results)

