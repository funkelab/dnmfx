import numpy as np
import zarr
import random
import cv2
import glob
import os
import re
from sklearn.datasets import make_blobs

class Parameters():

    def __init__(self, image_size, cell_size, num_cells, num_frames,
            random_seeds, trace_parameters, cell_centers, noise_magnitudes,
            home_path):

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
            random_seeds, trace_parameters, cell_centers, noise_magnitudes,
            home_path)
    parameters.data_path = os.path.join(parameters.home_path, "ground_truth")
    set_seeds(parameters)
    make_directories(parameters)
    save_cell_centers(parameters)
    traces = generate_traces(parameters)
    cell_coordinates = generate_cells(parameters)
    background = generate_background(parameters)
    assemble_video(parameters, traces, cell_coordinates, background)

def make_directories(parameters):
    
    '''
    Here shows the directories to be created for storing our Synthetic Dataset:

    └── reconstruction/ or ground_truth/
        ├── background/
        ├── cells/
        ├── centers/
        ├── noises/
        ├── components/
        ├── traces/
        ├── videos/
        └── ensemble/
    '''
    
    object_names = ['background', 'cells', 'centers', 'noises', 'components',
            'traces', 'ensemble', 'videos']
    
    reconstruction_path = f"{parameters.home_path}/reconstruction"
    ground_truth_path = f"{parameters.home_path}/ground_truth"

    ground_truth_store = zarr.DirectoryStore(ground_truth_path)
    reconstruction_store = zarr.DirectoryStore(reconstruction_path)
    self.ground_truth_data = zarr.group(store=ground_truth_store,
            overwrite=True)
    self.reconstructed_data = zarr.group(store=reconstruction_store,
            overwrite=True)

    for object_name in object_names:
        self.ground_truth_data.create_groups(object_name)
        self.reconstructed_data.create_groups(object_name)

def assemble_video(parameters, traces, cell_coordinates, background):

    image_array = []
    cell_array = []

    image_zarr = np.zeros( (parameters.num_frames, parameters.image_size,
        parameters.image_size) )
    cell_zarr = np.zeros( (parameters.num_frames, parameters.image_size,
        parameters.image_size) )

    # save all the cell components as ground truth
    for ci, cell in enumerate(cell_coordinates):

        cell_component = np.zeros(parameters.image_size**2)
        cell_indices = [int(i * parameters.image_size + j) for i, j in cell]
        np.put(cell_component, cell_indices, [255]*len(cell_indices))

        print(f'Saving cell frame{ci}.jpg and .zarr')  
        cell_component_path = f"{parameters.data_path}/components"
        cell_component = cell_component.reshape(parameters.image_size,
                parameters.image_size)
        cv2.imwrite(f"{cell_component_path}/jpg/component{ci}.jpg",
                cell_component)
        zarr.save(f"{cell_component_path}/zarr/component{ci}.zarr",
                zarr.array(cell_component))

    # assemble cell image at each frame
    for current_frame in range(parameters.num_frames):
        print(f"Assembling frame {current_frame}")
        assemble_frame(parameters, traces, cell_coordinates, current_frame, \
                background)
    
    ensemble_path = f"{parameters.data_path}/ensemble"
    cell_path = f"{parameters.data_path}/cells"

    for fname in sorted(glob.glob(f"{ensemble_path}/jpg/*.jpg"), \
            key=lambda x:float(re.findall("(\d+)",x)[0])):
        image_array.append(cv2.imread(fname))
    
    for fname in sorted(glob.glob(f"{ensemble_path}/zarr/*.zarr"), \
            key=lambda x:float(re.findall("(\d+)",x)[0])):
        current_frame = int(re.findall(r'\d+', fname)[0])
        image_zarr[current_frame, :, :] = zarr.load(fname)

    for fname in glob.glob(f"{cell_path}/jpg/*.jpg"):
        cell_array.append(cv2.imread(fname))

    for fname in sorted(glob.glob(f"{cell_path}/zarr/*.zarr"), \
        key=lambda x:float(re.findall("(\d+)",x)[0])):
        current_frame = int(re.findall(r'\d+', fname)[0])
        image_zarr[current_frame, :, :] = zarr.load(fname)
    
    video_path = f"{parameters.data_path}/videos"
    
    zarr.save(f"{video_path}/ensemble.zarr", image_zarr)
    zarr.save(f"{video_path}/cells.zarr", cell_zarr)

    ensemble_video = cv2.VideoWriter(f'{video_path}/ensemble.avi', 
            cv2.VideoWriter_fourcc('M','J','P','G'), 10,
            (parameters.image_size, parameters.image_size))
    cell_video = cv2.VideoWriter(f'{video_path}/cells.avi',
            cv2.VideoWriter_fourcc('M','J','P','G'), 10,
            (parameters.image_size, parameters.image_size))

    for i in range(len(image_array)):
        ensemble_video.write(image_array[i])
        cell_video.write(cell_array[i])

    ensemble_video.release()
    cell_video.release()

def assemble_frame(parameters, traces, cell_coordinates, current_frame,
                background):
    
    cell_intensities = traces[:, current_frame]
    calcium_image = np.copy(background)
    cell_image = np.array([255] * parameters.image_size**2)

    for ci, cell in enumerate(cell_coordinates):
        
        cell_indices = [int(i * parameters.image_size + j) for i, j in cell]
        color = int(cell_intensities[ci] * 256)
        # create image of cells only
        np.put(cell_image, cell_indices, [color]*len(cell_indices))
        # overlay cell intensities on top of background
        calcium_image[cell_indices] += color

    extra_noise = np.array([abs(np.random.normal(10, \
        parameters.extra_noise_mag)) for b in calcium_image])
    # overlay noise on top of cells + background
    calcium_image = np.array([calcium_image[i] + extra_noise[i] for i in \
        range(parameters.image_size**2)])

    save_object_frame(parameters, current_frame, "cells", cell_image)
    save_object_frame(parameters, current_frame, "noises", extra_noise)
    save_object_frame(parameters, current_frame, "ensemble", calcium_image)

def save_object_frame(parameters, current_frame, object_name, image_array):
    
    object_path = f"{parameters.data_path}/{object_name}"
    image_array = image_array.reshape(parameters.image_size,
            parameters.image_size)

    cv2.imwrite(f"{object_path}/jpg/frame{current_frame}.jpg", image_array)
    zarr.save(f"{object_path}/zarr/frame{current_frame}.zarr",
            zarr.array(image_array))

def generate_background(parameters):

    background = np.array([random.randint(0, int(256*parameters.bkg_noise_mag)) \
                            for i in range(parameters.image_size**2)])

    background_reshaped = background.reshape(parameters.image_size,
            parameters.image_size)
    cv2.imwrite(f"{parameters.data_path}/background/jpg/background.jpg",
            background_reshaped)
    zarr.save(f"{parameters.data_path}/background/zarr/background.zarr",
                zarr.array(background_reshaped))
    return background

def generate_traces(parameters):
    
    "Generate trace for ALL cells"
    
    all_traces = np.array([])
    for i in range(1, parameters.num_cells+1):

        cell_trace = generate_trace(parameters).reshape(1,-1)
        all_traces = np.append(cell_trace, all_traces).reshape(i, -1)

    trace_path = f"{parameters.data_path}/traces/zarr/traces.zarr"
    zarr.save(trace_path, all_traces)
    print(f'Cell trace saved!')

    return all_traces

def generate_trace(parameters):
    
    "Generate trace for ONE cell"
    
    s = np.random.poisson(parameters.possion_lam, parameters.num_frames)
    t = np.random.normal(parameters.gauss_mu, parameters.gauss_sigma,
            parameters.num_frames)
    signal = s*t
    signal[signal < 0] = 0.0 # smoothing
    signal[signal > 1.0] = 1.0
    
    return signal

def generate_cells(parameters):

    "Generates positions of cell coverage in a 512x512 coordinate"

    all_cell_coordinates = []
    for c in range(len(parameters.cell_centers)):
        
        X, _ = make_blobs( n_samples = int(parameters.cell_size**2),
                cluster_std=8, center_box=(parameters.cell_size/2,
                    parameters.cell_size/2) )
        cell_blocks = list(zip(X[:, 0].astype(int), X[:, 1].astype(int)))
        # reindexing to get a single cell coordinates on a 512x512 image
        single_cell_coordinates = [ ( parameters.cell_centers[c][0] * \
            parameters.cell_size + cb[0], parameters.cell_centers[c][1] * \
            parameters.cell_size + cb[1] ) for cb in cell_blocks]
        all_cell_coordinates.append(single_cell_coordinates)

    return all_cell_coordinates

def set_seeds(parameters):

    np.random.seed(parameters.np_seed)
    random.seed(parameters.random_seed)

def save_cell_centers(parameters):

    centers = np.array([(r*parameters.cell_size, c*parameters.cell_size) \
            for (r, c) in parameters.cell_centers])
    centers_path = f"{parameters.data_path}/centers/zarr/centers.zarr"
    zarr.save(centers_path, centers)
    print(f'Cell centers saved!')

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
    noise_magnitudes = {"background": 0.4, "extra":10}
    home_path = "/home/luk@hhmi.org/DNMFX/dnmfx"

    # hard-coded cell positions with three overlapping cells
    overlap_cell_centers = [(0, 1.6), (5, 3.6), (7, 5.7)]
    other_cell_centers = [(0, 1), (1, 3), (3, 0), (5, 3), (6,2), (6, 5), (7, 5)]
    cell_centers = overlap_cell_centers + other_cell_centers
    synthesize(img_size, cell_size, num_cells, num_frames, random_seeds,
             trace_params, cell_centers, noise_magnitudes, home_path)
