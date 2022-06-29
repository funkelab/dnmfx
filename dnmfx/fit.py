import numpy as np
from nmfx import *
import cv2
import glob
from matplotlib import pyplot as plt


def fit(ds_name, image_size, save_results=True, max_iteration=100,
        min_loss=1e-3, batch_size=10, step_size=1e-1, l1_W=0, num_frames=100,
        num_components=11, print_iter=10):

    params = parameters.Parameters()
    params.max_iteration = max_iteration
    params.min_loss = min_loss
    # Initialize H, W, X
    H = np.random.randn(num_frames, num_components)
    W = np.random.randn(num_components, img_size**2)
    initial_values = {"H": H, "W": W}
    X = np.zeros(img_size**2).reshape(1, -1)

    for fname in glob.glob(f"{ds_name}/*.jpg"):
        img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE).reshape(1, -1)
        X = np.concatenate((X, img), axis=0)
    X = X[1:, :]

    H, W, log = nmf(X, num_components, params, print_iter, initial_values)
    
    if save_results:
        save_reconstruction(H, W, num_components)

    return {"H": H, "W": W, "log": log}

def save_reconstruction(H, W, num_components):
    
    if not os.path.exists("reconstruct"):
        os.mkdir("reconstruct")

    for i in range(num_components):
        plt.imsave(f"reconstruct/cell{i}.jpg", W[i, :].reshape(img_size, img_size),
                cmap=plt.cm.bone)
        plt.plot(H[:, i])
        plt.savefig(f"reconstruct/trace{i}.jpg")
        plt.close()

    np.save("reconstruct/traces.npy", H)

if __name__ == "__main__":
    
    reconstruction = fit("ground_truth/all", 512)

