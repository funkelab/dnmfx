import numpy as np
import random
import cv2
import glob
import os
from random import sample
from scipy.stats import poisson, norm
from PIL import Image
from sklearn.datasets import make_blobs

class Parameters():

    def __init__(self, img_size, cell_size, num_cells, num_frames, random_seeds,
            trace_params, cell_centers, noise_magnitudes, ds_path):

        self.img_size = img_size
        self.cell_size = cell_size
        self.num_cells = num_cells
        self.num_frames = num_frames

        self.ds_path = ds_path
        self.img_dir = None

        self.np_seed = random_seeds['numpy']
        self.random_seed = random_seeds["random"]
        
        self.possion_lam = trace_params['possion_lam']
        self.gauss_mu = trace_params['gauss_mu']
        self.gauss_sigma = trace_params['gauss_sigma']

        self.overlap_cell_centers = cell_centers['overlap']
        self.other_cell_centers = cell_centers['other']
        
        self.bkg_noise_mag = noise_magnitudes['bkg']
        self.extra_noise_mag = noise_magnitudes['extra']

def synthesize(img_size, cell_size, num_cells, num_frames, random_seeds,
        trace_params, cell_centers, noise_magnitudes, ds_path):

    params = Parameters(img_size, cell_size, num_cells, num_frames,
            random_seeds, trace_params, cell_centers, noise_magnitudes, ds_path)
    
    params.img_dir = os.path.join(params.ds_path, "grd_truth")
    dir_names = [f"{params.img_dir}", f"{params.img_dir}/cells", 
            f"{params.img_dir}/background", f"{params.img_dir}/singles", 
            f"{params.img_dir}/noises", f"{params.img_dir}/traces", 
            f"{params.img_dir}/all"]

    for dir_name in dir_names:
        if os.path.exists(dir_name) == False:
            os.mkdir(dir_name)

    traces = generate_traces(params)
    with open(f"{params.img_dir}/traces/traces.npy", "wb") as f:
        np.save(f, traces)
    print(f'traces has shape {traces.shape}')

    other_cells, overlap_cells = generate_cells(params)
    background = generate_background(params)
    assemble_video(params, traces, other_cells, overlap_cells, background)

def assemble_video(params, traces, other_cells, overlap_cells, background):

    img_array = []
    cell_array = []

    # save all the cell components as ground truth
    for ci, cell in enumerate(other_cells + overlap_cells):

        single_cell_img = np.zeros(params.img_size**2)
        cell_indices = [int(i * params.img_size + j) for i, j in cell]
        np.put(single_cell_img, cell_indices, [255]*len(cell_indices))
        print(f'Saving cell {ci}')
        cell_img_dir = f"{params.img_dir}/singles"
        
        if not cv2.imwrite(f"{cell_img_dir}/cell{ci}.jpg",
                single_cell_img.reshape(params.img_size, params.img_size)):
            raise Exception("Cell image could not be written")
        else:
            cv2.imwrite(f"{cell_img_dir}/cell{ci}.jpg",
                    single_cell_img.reshape(params.img_size, params.img_size))

    # assemble cell image at each frame
    for curr_frame in range(params.num_frames):
        print(f"Processing frame {curr_frame}")
        assemble_img(params, traces, other_cells, overlap_cells, curr_frame, background)

    for fname in glob.glob(f"{params.img_dir}/all/*.jpg"):
        img_array.append( cv2.imread(fname) )
    
    for fname in glob.glob(f"{params.img_dir}/cells/*.jpg"):
        cell_array.append(cv2.imread(fname))

    out_all = cv2.VideoWriter('all.avi', \
            cv2.VideoWriter_fourcc('M','J','P','G'), 10, (img_size, img_size))
    
    out_cell = cv2.VideoWriter('cells.avi', \
            cv2.VideoWriter_fourcc('M','J','P','G'), 10, (img_size, img_size))

    for i in range(len(img_array)):
        out_all.write(img_array[i])
        out_cell.write(cell_array[i])

    out_all.release()
    out_cell.release()

def assemble_img(params, traces, other_cells, overlap_cells, curr_frame, background):

    cell_intensities = traces[:, curr_frame]
    calcium_img = np.copy(background)

    #cell_img = np.zeros(img_size**2)
    cell_img = np.array([255] * params.img_size**2)

    for ci, cell in enumerate(other_cells + overlap_cells):

        cell_indices = [int(i * params.img_size + j) for i, j in cell]
        color = int(cell_intensities[ci] * 256)
        
        # create pure cell imgs
        np.put(cell_img, cell_indices, [color]*len(cell_indices))

        # overlay cell intensities on top of background
        if cell in overlap_cells:
            calcium_img[cell_indices] += color
        else:
            np.put(calcium_img, cell_indices, [color]*len(cell_indices) )
    
    extra_noise = np.array([abs(np.random.normal(10, params.extra_noise_mag)) for b in calcium_img])
    # overlay noise on top of cells + background
    calcium_img = np.array([calcium_img[i] + extra_noise[i] for i in \
        range(params.img_size**2)])

    save_img(params, curr_frame, "cells", cell_img)
    save_img(params, curr_frame, "noises", extra_noise)
    save_img(params, curr_frame, "all", calcium_img)

def save_img(params, curr_frame, obj, img_array):
    cv2.imwrite(f"{params.img_dir}/{obj}/frame{curr_frame}.jpg",
            img_array.reshape(params.img_size, params.img_size))

def generate_background(params):
    
    background = np.array([random.randint(0, int(256*params.bkg_noise_mag)) \
                            for i in range(params.img_size**2)])
    cv2.imwrite(f"{params.img_dir}/background/background.jpg", 
                background.reshape(params.img_size, params.img_size))

    return background

def generate_traces(params):
    
    "Generate trace for ALL cells"

    all_traces = np.array([])

    for i in range(1, params.num_cells+1):

        cell_trace = generate_trace(params).reshape(1,-1)
        all_traces = np.append(cell_trace, all_traces).reshape(i, -1)
    
    return all_traces

def generate_trace(params):
    
    "Generate trace for ONE cell"
    
    s = np.random.poisson(params.possion_lam, params.num_frames)
    t = np.random.normal(params.gauss_mu, params.gauss_sigma, params.num_frames)
    
    signal = s*t
    
    signal[signal < 0] = 0.0 # smoothing
    signal[signal > 1.0] = 1.0
    
    return signal

def generate_cells(params):

    "Generates positions of cell coverage in a 512x512 coordinate"

    other_cells, overlap_cells = [], []
    
    for c in range(len(params.other_cell_centers)):
        
        X, _ = make_blobs(n_samples = int(params.cell_size**2), cluster_std=8, 
                center_box=(params.cell_size/2, params.cell_size/2))
        cell_blocks = list(zip(X[:, 0].astype(int), X[:, 1].astype(int)))
        
        # reindexing to get a single cell coordinates on a 512x512 image
        other_cell_coords = [ ( params.other_cell_centers[c][0] * \
            params.cell_size + cb[0], params.other_cell_centers[c][1] * \
            params.cell_size + cb[1] ) for cb in cell_blocks]
        other_cells.append(other_cell_coords)
    
    for c in range(len(params.overlap_cell_centers)):

        X, _ = make_blobs(n_samples = int(params.cell_size**2), cluster_std=8,
                center_box=(params.cell_size/2, params.cell_size/2))
        cell_blocks = list(zip(X[:, 0].astype(int), X[:, 1].astype(int)))

        overlap_cell_coords = [ ( params.overlap_cell_centers[c][0] * \
            params.cell_size + cb[0], params.overlap_cell_centers[c][1] * \
            params.cell_size + cb[1] ) for cb in cell_blocks]
        overlap_cells.append(overlap_cell_coords)

    return other_cells, overlap_cells

def set_seeds(params):

    np.random.seed(params.np_seed)
    random.seed(params.random_seed)

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
    noise_magnitudes = {"bkg": 0.4, "extra":10}
    ds_path = ""

    # hard-coded cell positions with three overlapping cells
    overlap_cell_centers = [(0, 1.6), (5, 3.6), (7, 5.7)]
    other_cell_centers = [(0, 1), (1, 3), (3, 0), (5, 3), (6,2), (6, 5), (7, 5)]
    cell_centers = {"overlap": overlap_cell_centers, "other": other_cell_centers}

    synthesize(img_size, cell_size, num_cells, num_frames, random_seeds,
             trace_params, cell_centers, noise_magnitudes, ds_path)
