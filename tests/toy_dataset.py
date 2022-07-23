from dnmfx import Dataset
from skimage.filters import gaussian
import funlib.geometry as fg
import numpy as np
import zarr


class Parameters():

    def __init__(
            self,
            image_size,
            cell_size,
            num_frames,
            cell_centers,
            trace_parameters,
            noise_magnitudes,
            random_seed=None):

        self.image_size = image_size
        self.cell_size = cell_size
        self.num_cells = len(cell_centers)
        self.num_frames = num_frames

        self.random_seed = random_seed

        self.possion_lam = trace_parameters['possion_lam']
        self.gauss_mu = trace_parameters['gauss_mu']
        self.gauss_sigma = trace_parameters['gauss_sigma']

        self.cell_centers = cell_centers
        self.bg_noise_mag = noise_magnitudes['background']
        self.extra_noise_mag = noise_magnitudes['extra']


def create_synthetic_dataset(parameters):

    np.random.seed(parameters.random_seed)

    bounding_boxes = generate_bounding_boxes(parameters)
    traces = generate_traces(parameters)
    components = generate_components(parameters)
    background = generate_background(parameters)
    noises = generate_noises(parameters)

    return Dataset(bounding_boxes, components, traces, background, noises)


def generate_traces(parameters):

    return np.stack([
        generate_trace(parameters)
        for _ in range(parameters.num_cells)
    ])


def generate_components(parameters):

    return np.stack([
        generate_component(parameters)
        for _ in range(parameters.num_cells)
    ])


def generate_component(parameters):

    size = parameters.cell_size

    component = np.zeros((size, size), dtype=np.float32)

    # set center of component to 1
    component[size//2, size//2] = 1.0

    # blur to get a blob
    component = gaussian(component, sigma=size/4)

    # multiply with random noise
    component *= 0.5 + 0.5 * np.random.uniform(size=(size, size))

    # normalize such that brightest pixel is 1
    component /= np.max(component)

    return component


def generate_bounding_boxes(parameters):

    return [
        fg.Roi(
            (
                y - parameters.cell_size/2,
                x - parameters.cell_size/2
            ),
            (
                parameters.cell_size,
                parameters.cell_size
            )
        )
        for x, y in parameters.cell_centers
    ]


def generate_background(parameters):

    size = parameters.image_size

    return np.random.uniform(size=(size, size)) * parameters.bg_noise_mag


def generate_noises(parameters):

    size = parameters.image_size

    return np.array([
        np.random.uniform(size=(size, size)) * parameters.extra_noise_mag
        for _ in range(parameters.num_frames)
    ])


def generate_trace(parameters):

    s = np.random.poisson(parameters.possion_lam, parameters.num_frames)
    t = np.random.normal(parameters.gauss_mu, parameters.gauss_sigma,
                         parameters.num_frames)
    signal = s*t
    signal[signal < 0] = 0.0    # smoothing
    signal[signal > 1.0] = 1.0

    return signal
