import numpy as np
import zarr
import random
import os
from sklearn.datasets import make_blobs


class Parameters():

    def __init__(self, image_size, cell_size, num_cells, num_frames,
                 random_seeds, trace_parameters, cell_centers,
                 noise_magnitudes, home_path):

        self.image_size = image_size
        self.cell_size = cell_size
        self.num_cells = num_cells
        self.num_frames = num_frames

        self.home_path = home_path
        self.data_path = None
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


def synthesize(image_size, cell_size, num_cells, num_frames, random_seeds,
               trace_parameters, cell_centers, noise_magnitudes, home_path):

    parameters = Parameters(image_size, cell_size, num_cells, num_frames,
                            random_seeds, trace_parameters, cell_centers,
                            noise_magnitudes, home_path)
    parameters.data_path = os.path.join(parameters.home_path, "ground_truth")
    set_seeds(parameters)
    make_directories(parameters)

    centers = np.array([(r*parameters.cell_size, c*parameters.cell_size) for
                        (r, c) in parameters.cell_centers])
    write_object_to_zarr(parameters.ground_truth_data, "centers", centers)

    traces = generate_traces(parameters)
    write_object_to_zarr(parameters.ground_truth_data, "traces", traces)

    cell_coordinates = generate_cells(parameters)
    save_cell_centers(parameters)
    save_cell_components(parameters, cell_coordinates)

    background = generate_background(parameters)
    background_reshaped = background.reshape(parameters.image_size,
                                             parameters.image_size)
    write_object_to_zarr(parameters.ground_truth_data, "background",
                         background_reshaped)

    cells = zarr.group(store=f"{parameters.data_path}/cells", overwrite=True)
    ensemble = zarr.group(store=f"{parameters.data_path}/ensemble",
                          overwrite=True)
    ensemble_sequence, cell_sequence = assemble_sequence(parameters, traces,
                                                         cell_coordinates,
                                                         background, ensemble,
                                                         cells)
    write_object_to_zarr(ensemble, "sequence", ensemble_sequence)
    write_object_to_zarr(cells, "sequence", cell_sequence)


def make_directories(parameters):
    '''
    Here shows the Zarr groups to be created for storing our Synthetic Dataset:
    └── reconstruction/ or ground_truth/
        ├── background/
        ├── cells/
        ├── centers/
        ├── noises/
        ├── components/
        ├── traces/
        └── ensemble/
    '''

    object_names = ['background', 'cells', 'centers', 'noises', 'components',
                    'traces', 'ensemble']

    reconstruction_path = f"{parameters.home_path}/reconstruction"
    parameters.data_path = f"{parameters.home_path}/ground_truth"

    ground_truth_store = zarr.DirectoryStore(parameters.data_path)
    reconstruction_store = zarr.DirectoryStore(reconstruction_path)
    parameters.ground_truth_data = zarr.group(store=ground_truth_store,
                                              overwrite=True)
    parameters.reconstructed_data = zarr.group(store=reconstruction_store,
                                               overwrite=True)

    for object_name in object_names:
        parameters.ground_truth_data.create_group(object_name)
        parameters.reconstructed_data.create_group(object_name)


def save_cell_centers(parameters):

    cell_centers = np.array([(r*parameters.cell_size, c*parameters.cell_size)
                            for (r, c) in parameters.cell_centers])
    write_object_to_zarr(parameters.ground_truth_data, "centers", cell_centers)
    print('cell centers saved!')


def save_cell_components(parameters, cell_coordinates):

    components = zarr.group(store=f"{parameters.data_path}/components",
                            overwrite=True)
    # save all the cell components as ground truth
    for ci, cell in enumerate(cell_coordinates):
        cell_component = np.zeros(parameters.image_size**2)
        cell_indices = [int(i * parameters.image_size + j) for i, j in cell]
        np.put(cell_component, cell_indices, [255]*len(cell_indices))
        cell_component = cell_component.reshape(parameters.image_size,
                                                parameters.image_size)
        write_object_to_zarr(components, f"component{ci}",
                             zarr.array(cell_component))
        print(f'cell frame{ci}.zarr saved')


def assemble_sequence(parameters, traces, cell_coordinates, background,
                      ensemble, cells):

    ensemble_sequence = np.zeros((parameters.num_frames, parameters.image_size,
                                  parameters.image_size))
    cell_sequence = np.zeros((parameters.num_frames, parameters.image_size,
                              parameters.image_size))

    # assemble cell at each frame
    for current_frame in range(parameters.num_frames):
        ensemble_frame, cell_frame = assemble_frame(parameters, traces,
                                                    cell_coordinates,
                                                    current_frame, background,
                                                    cells, ensemble)
        cell_frame = cell_frame.reshape(parameters.image_size,
                                        parameters.image_size)
        ensemble_frame = ensemble_frame.reshape(parameters.image_size,
                                                parameters.image_size)
        write_object_to_zarr(cells, f"frame{current_frame}",
                             zarr.array(cell_frame))
        write_object_to_zarr(ensemble, f"frame{current_frame}",
                             zarr.array(ensemble_frame))
        print(f"frame {current_frame} assembled")
        ensemble_sequence[current_frame, :, :] = zarr.array(ensemble_frame)
        cell_sequence[current_frame, :, :] = zarr.array(cell_frame)

    return ensemble_sequence, cell_sequence


def assemble_frame(parameters, traces, cell_coordinates, current_frame,
                   background, cells, ensemble):

    cell_intensities = traces[:, current_frame]
    ensemble_frame = np.copy(background)
    cell_frame = np.array([255] * parameters.image_size**2)

    for ci, cell in enumerate(cell_coordinates):
        cell_indices = [int(i * parameters.image_size + j) for i, j in cell]
        color = int(cell_intensities[ci] * 256)
        # create image of cells only
        np.put(cell_frame, cell_indices, [color]*len(cell_indices))
        # create calcium image
        ensemble_frame[cell_indices] += color

    extra_noise = np.array([abs(np.random.normal(10,
                           parameters.extra_noise_mag)) for b in
                           ensemble_frame])
    ensemble_frame = np.array([ensemble_frame[i] + extra_noise[i] for i in
                               range(parameters.image_size**2)])

    return ensemble_frame, cell_frame


def generate_background(parameters):

    background = np.array([random.randint(0, int(256*parameters.bkg_noise_mag))
                           for i in range(parameters.image_size**2)])

    return background


def generate_traces(parameters):

    "Generate trace for ALL cells"
    all_traces = np.array([])
    for i in range(1, parameters.num_cells+1):
        cell_trace = generate_trace(parameters).reshape(1, -1)
        all_traces = np.append(cell_trace, all_traces).reshape(i, -1)

    return all_traces


def generate_trace(parameters):

    "Generate trace for ONE cell"
    s = np.random.poisson(parameters.possion_lam, parameters.num_frames)
    t = np.random.normal(parameters.gauss_mu, parameters.gauss_sigma,
                         parameters.num_frames)
    signal = s*t
    signal[signal < 0] = 0.0    # smoothing
    signal[signal > 1.0] = 1.0

    return signal


def generate_cells(parameters):

    all_cell_coordinates = []
    for c in range(len(parameters.cell_centers)):
        X, _ = make_blobs(n_samples=int(parameters.cell_size**2),
                          cluster_std=8, center_box=(parameters.cell_size/2,
                          parameters.cell_size/2))
        cell_blocks = list(zip(X[:, 0].astype(int), X[:, 1].astype(int)))
        # reindexing to get a single cell coordinates on a 512x512 image
        single_cell_coordinates = [(parameters.cell_centers[c][0] *
                                   parameters.cell_size + cb[0],
                                   parameters.cell_centers[c][1] *
                                   parameters.cell_size + cb[1]) for cb in
                                   cell_blocks]
        all_cell_coordinates.append(single_cell_coordinates)

    return all_cell_coordinates


def write_object_to_zarr(zarr_group, object_name, object_data):
    zarr_group[object_name] = object_data


def set_seeds(parameters):
    np.random.seed(parameters.np_seed)
    random.seed(parameters.random_seed)


if __name__ == "__main__":
    random_seeds = {"numpy": 88, "random": 2001}
    trace_params = {
                        "possion_lam": 2,
                        "gauss_mu": 0,
                        "gauss_sigma": 0.1
                    }
    img_size = 512
    cell_size = 64
    num_cells = 10
    num_frames = 100

    # background noise magnitude ranges from 0 (no noise) to 1 (very noisy)
    # extra noise magnitude >= 0 with 0 being no extra noise
    noise_magnitudes = {"background": 0.4, "extra": 10}
    home_path = "/home/luk@hhmi.org/DNMFX/dnmfx"

    # hard-coded cell positions with three overlapping cells
    overlap_cell_centers = [(0, 1.6), (5, 3.6), (7, 5.7)]
    other_cell_centers = [(0, 1), (1, 3), (3, 0), (5, 3), (6, 2), (6, 5),
                          (7, 5)]
    cell_centers = overlap_cell_centers + other_cell_centers
    synthesize(img_size, cell_size, num_cells, num_frames, random_seeds,
               trace_params, cell_centers, noise_magnitudes, home_path)
