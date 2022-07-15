import numpy as np
import zarr
import random
from sklearn.datasets import make_blobs


class Dataset():

    def __init__(self, ensembled_components, singular_components, traces,
                 background, noises, ensemble, centers):
        self.ensembled_components = ensembled_components
        self.singular_components = singular_components
        self.traces = traces
        self.background = background
        self.noises = noises
        self.ensemble = ensemble
        self.centers = centers


class Parameters():

    def __init__(self, image_size, cell_size, num_cells, num_frames,
                 random_seeds, trace_parameters, cell_centers,
                 noise_magnitudes):

        self.image_size = image_size
        self.cell_size = cell_size
        self.num_cells = num_cells
        self.num_frames = num_frames

        self.ground_truth_data = None
        self.reconstructed_data = None

        self.np_seed = random_seeds['numpy']
        self.random_seed = random_seeds["random"]

        self.possion_lam = trace_parameters['possion_lam']
        self.gauss_mu = trace_parameters['gauss_mu']
        self.gauss_sigma = trace_parameters['gauss_sigma']

        self.cell_centers = cell_centers
        self.bkg_noise_mag = noise_magnitudes['background']
        self.extra_noise_mag = noise_magnitudes['extra']


def set_parameters(image_size, cell_size, num_cells, num_frames,
                   random_seeds, trace_parameters, cell_centers,
                   noise_magnitudes):

    return Parameters(image_size, cell_size, num_cells, num_frames,
                      random_seeds, trace_parameters, cell_centers,
                      noise_magnitudes)


def create_synthetic_dataset(parameters):

    centers = np.array([(r*parameters.cell_size, c*parameters.cell_size) for
                        (r, c) in parameters.cell_centers])
    traces = generate_traces(parameters)

    component_coordinates = generate_component_coordinates(parameters)
    singular_components = generate_singular_components(parameters, traces,
                                                       component_coordinates)
    ensembled_components = generate_ensembled_components(parameters, traces,
                                                         component_coordinates)
    background = generate_background(parameters)
    noises = generate_noises(parameters)
    ensemble = generate_ensemble(parameters, noises, background,
                                 ensembled_components)

    # after creating all elements..
    dataset = Dataset(singular_components, ensembled_components, traces,
                      background, noises, ensemble, centers)
    return dataset


def generate_traces(parameters):

    all_traces = np.array([])
    for i in range(1, parameters.num_cells+1):
        cell_trace = generate_trace(parameters).reshape(1, -1)
        all_traces = np.append(cell_trace, all_traces).reshape(i, -1)

    return all_traces


def generate_component_coordinates(parameters):

    component_coordinates = []

    for c in range(len(parameters.cell_centers)):
        X, _ = make_blobs(n_samples=int(parameters.cell_size**2),
                          cluster_std=8, center_box=(parameters.cell_size/2,
                          parameters.cell_size/2))
        cell_blocks = list(zip(X[:, 0].astype(int), X[:, 1].astype(int)))
        # reindexing to get a single cell coordinates on a 512x512 image
        single_component_coordinates = [(parameters.cell_centers[c][0] *
                                   parameters.cell_size + cb[0],
                                   parameters.cell_centers[c][1] *
                                   parameters.cell_size + cb[1]) for cb in
                                   cell_blocks]
        component_coordinates.append(single_component_coordinates)

    return component_coordinates


def generate_background(parameters):

    return np.array([random.randint(0, int(256*parameters.bkg_noise_mag))
                     for i in range(parameters.image_size**2)])


def generate_noises(parameters):

    return [np.array([abs(np.random.normal(10, parameters.extra_noise_mag))
                       for i in range(parameters.image_size**2)])
            for frame_index in range(parameters.num_frames)]


def generate_singular_components(parameters, traces, component_coordinates):

    '''Create a list of components of shape (num_frames, component_dimension)
    '''

    singular_components = []
    for component_index, coordinates in enumerate(component_coordinates):
        component_indices = [int(i * parameters.image_size + j)
                             for i, j in coordinates]
        component = np.zeros((parameters.num_frames, parameters.image_size**2))
        for frame_index in range(parameters.num_frames):
            color = int(traces[:, frame_index][component_index] * 256)
            component[frame_index][component_indices] += color
        singular_components.append(component)

    return singular_components


def generate_ensembled_components(parameters, traces, component_coordinates):

    '''Create a list of length number of frames that contains component of shape
    (component_dimension, ), i.e. a list of all components at each time step
    '''
    ensembled_components = []
    for frame_index in range(parameters.num_frames):
        component_frame = np.zeros(parameters.image_size**2)
        for component_index, component in enumerate(component_coordinates):
            component_indices = [int(i * parameters.image_size + j)
                                 for i, j in component]
            color = int(traces[:, frame_index][component_index] * 256)
            component_frame[component_indices] += color
        ensembled_components.append(component_frame)

    return ensembled_components


def generate_ensemble(parameters, noises, background, ensembled_components):

    return [np.array([background[i] + noises[t][i] + ensembled_components[t][i]
                      for i in range(parameters.image_size**2)])
            for t in range(parameters.num_frames)]


def generate_trace(parameters):

    s = np.random.poisson(parameters.possion_lam, parameters.num_frames)
    t = np.random.normal(parameters.gauss_mu, parameters.gauss_sigma,
                         parameters.num_frames)
    signal = s*t
    signal[signal < 0] = 0.0    # smoothing
    signal[signal > 1.0] = 1.0

    return signal

