import zarr
import numpy as np
from nmfx import parameters, nmf


class FittingParameters():

    def __init__(self, home_path, save_reconstruction, max_iteration, min_loss,
                 batch_size, step_size, l1_W, num_components,
                 print_iterations):

        self.nmf_parameters = parameters.Parameters()
        self.nmf_parameters.max_iter = max_iteration
        self.nmf_parameters.min_loss = min_loss  # min_diff
        self.nmf_parameters.batch_size = batch_size
        self.nmf_parameters.step_size = step_size
        self.nmf_parameters.l1_W = l1_W

        self.image_size = None
        self.num_frames = None
        self.home_path = home_path
        self.save_reconstruction = save_reconstruction
        self.num_components = num_components
        self.print_iterations = print_iterations
        self.initial_values = None


def fit(home_path, save_reconstruction, max_iteration, min_loss, batch_size,
        step_size, l1_W, num_components, print_iterations):

    fitting_parameters = FittingParameters(home_path, save_reconstruction,
                                           max_iteration, min_loss, batch_size,
                                           step_size, l1_W, num_components,
                                           print_iterations)
    data_path = f"{home_path}/ground_truth/ensemble/sequence.zarr"
    X = zarr.load(data_path)
    fitting_parameters.num_frames, fitting_parameters.image_size, _ = X.shape
    X = X.reshape(fitting_parameters.num_frames,
                  fitting_parameters.image_size**2)
    # Initialize H, W, X
    H = np.random.randn(fitting_parameters.num_frames, num_components)
    W = np.random.randn(num_components, fitting_parameters.image_size**2)
    fitting_parameters.initial_values = {"H": H, "W": W}
    H, W, log = nmf(X, num_components, fitting_parameters.nmf_parameters,
                    fitting_parameters.print_iterations,
                    fitting_parameters.initial_values)

    if fitting_parameters.save_reconstruction:
        save_component_traces(H, W, num_components, fitting_parameters)

    return {"H": H, "W": W, "log": log}


def save_component_traces(H, W, num_components, fitting_parameters):

    components, traces = make_directories(fitting_parameters)
    for i in range(num_components):
        component = W[i, :].reshape(fitting_parameters.image_size,
                                    fitting_parameters.image_size)
        trace = H[:, i]
        # save reconstructed components
        components[f"component{i}.zarr"] = zarr.array(component)
        # save reconstructed traces
        traces[f"trace{i}.zarr"] = zarr.array(trace)


def make_directories(fitting_parameters):

    reconstruction_path = f"{fitting_parameters.home_path}/reconstruction"
    reconstruction_store = zarr.DirectoryStore(reconstruction_path)
    component_group = zarr.group(store=f"{reconstruction_path}/components",
                                 overwrite=True)
    trace_group = zarr.group(store=f"{reconstruction_path}/traces",
                             overwrite=True)

    return component_group, trace_group
