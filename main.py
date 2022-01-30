import diffuser_cam
if __name__ == '__main__':
    # Loads configuration parameters.
    config = diffuser_cam.load_config('config/default.json')

    diffuser_cam.load_data('img/psf.tif', 'img/measurement.tif', downsample_factor=config['downsample_factor'], show_im=True)
