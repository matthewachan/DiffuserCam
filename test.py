import base64
import diffuser_cam
import matplotlib.pyplot as plt
import cv2
import numpy as np

def base_test(config):
    # Loads configuration parameters.
    config = diffuser_cam.load_config(config)
    psf, data = diffuser_cam.load_data('img/psf.tif', 'img/measurement.tif', downsample_factor=config['downsample_factor'], show_im=False)

    # Get "ground-truth"
    # reconstruction = diffuser_cam.sgd(psf, data, n_iter=config['n_iter'], show_im=True)
    reconstruction = diffuser_cam.pnp_admm(psf, data, n_iter=config['n_iter'], rho=0.00001, show_im=True)

    psf, data = diffuser_cam.load_sim_data(reconstruction, psf, downsample_factor=1, show_im=False)
    # reconstruction = diffuser_cam.sgd(psf, data, n_iter=500, show_im=True)
    reconstruction = diffuser_cam.pnp_admm(psf, data, n_iter=config['n_iter'], rho=0.00000001, show_im=True)


def run_sim(config):
    # Loads configuration parameters.
    config = diffuser_cam.load_config(config)

    psf = cv2.imread('img/psf.tif', cv2.IMREAD_GRAYSCALE).astype(float)
    gt = cv2.imread('img/test/window.jpg', cv2.IMREAD_GRAYSCALE).astype(float)
    gt = cv2.resize(gt, psf.shape[::-1])

    psf, data = diffuser_cam.load_sim_data(gt, psf, config['downsample_factor'], config['noise_level'], show_im=False)

    downsample = diffuser_cam.utils.downsample(gt, config['downsample_factor'])
    plt.figure()
    plt.title('Ground truth (downsampled)')
    plt.imshow(downsample)

    # reconstruction = diffuser_cam.sgd(psf, data, n_iter=config['n_iter'], show_im=True)
    # reconstruction = diffuser_cam.tikhonov(psf, data, config['n_iter'], config['alpha'], show_im=True)
    reconstruction = diffuser_cam.pnp_admm(psf, data, n_iter=config['n_iter'], rho=config['rho'], show_im=True)

if __name__ == '__main__':
    config_file = 'config/default.json'
    # base_test(config_file)
    run_sim(config_file)