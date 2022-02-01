import diffuser_cam
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Loads configuration parameters.
    config = diffuser_cam.load_config('config/default.json')

    psf, data = diffuser_cam.load_data('img/psf.tif', 'img/measurement.tif', downsample_factor=config['downsample_factor'], show_im=False)

    # reconstruction = diffuser_cam.sgd(psf, data, n_iter=config['n_iter'], show_im=True)
    reconstruction = diffuser_cam.pnp_admm(psf, data, n_iter=config['n_iter'], rho=config['rho'], show_im=True)