import diffuser_cam
import matplotlib.pyplot as plt
import cv2
import json
import numpy as np

# Runs the reconstruction algorithm on real data.
def main():
    algo = diffuser_cam.Algorithm('config/algorithm.json')
    dataloader = diffuser_cam.DataLoader('config/real_data.json')
    psf, data = dataloader.load_real_data()
    algo.reconstruct(psf, data, show_im=True)

<<<<<<< Updated upstream
# Example script to evalute the algorithm reconstructions in simulation.
=======

def normalize(img):
    return 255 * (img - np.amin(img) / (np.amax(img) - np.amin(img)))

# Example script to evaluate the reconstruction algorithms in simulation.
>>>>>>> Stashed changes
def run_sim():
    algo = diffuser_cam.Algorithm('config/algorithm.json')
    dataloader = diffuser_cam.DataLoader('config/sim_data.json')
    psf, data, gt = dataloader.load_sim_data()
    plt.figure()
    plt.title('Ground truth (downsampled)')
    plt.imshow(gt)
    reconstruction = algo.reconstruct(psf, data, show_im=True)

"""Finds a local minimum hyperparameter value that maximizes the PSNR.

    Args:
        config_fname: The algorithm config fileto run.
        parameter: The hyperparameter to optimize.
        start: The start value of the hyperparameter.
        end: The end value of the hyperparameter.
        step: Amount to increment the hyperparameter by between iterations.
"""
def cross_validation(start, end, num_steps):
    dataloader = diffuser_cam.DataLoader('config/sim_data.json')
    config_fname = 'config/algorithm.json'
    with open(config_fname) as f:
        config = json.load(f)
        for alpha in np.arange(start, end, (end - start) / num_steps):
            config['alpha'] = alpha
            algo = diffuser_cam.Algorithm(config)
            psf, data, gt = dataloader.load_sim_data()
            reconstruction = algo.reconstruct(psf, data)
            print(alpha, cv2.PSNR(normalize(reconstruction), normalize(gt)))


if __name__ == '__main__':
    # main()
    # run_sim()
    cross_validation(0.01, 0.00000001, 100)