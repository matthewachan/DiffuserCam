"""
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
"""
import cv2
import json
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import typing

from diffuser_cam import utils

class DataLoader():
    """Loads and preprocesses PSFs, diffuser cam images, and ground truth images. The output of the DataLoader serves as input to the reconstruction algorithm. (See the Algorithm class for more details).

    The DataLoader configuration file should be in JSON format and can contain the following configuration parameters:
        psf: Path to the PSF image.
        measurement: Path to the diffuser cam measurement image. This parameter is only requird when calling load_real_data().
        ground_truth: Path to the ground-truth image. This parameter is only required when calling load_sim_data().
        downsample_factor: Floating point number specifying the amount to downsample the input PSF, diffuser cam, and grouth truth images.
        noise_level: Floating point number denoting how much white Gaussian noise to add to the simulated diffuser cam image. This parameter is only requird when calling load_sim_data().

    Attributes:
        downsample_factor: Amount to downsample the images.
        noise_level: Level of added white Gaussian noise used in simulated reconstructions.
        psf_fname: Path to the PSF image.
        gt_fname: Path to the ground truth image.
        measurement_fname: Path to the diffuser cam measurement image.
    """
    def __init__(self, config_fname: str) -> None:
        """Loads the JSON config file.
        
        Args:
            config_fname: Path to the JSON configuration file.
        """
        with open(config_fname) as f:
            config = json.load(f)
            self.psf_fname = config['psf']
            self.measurement_fname = config.get('measurement')
            self.gt_fname = config.get('ground_truth')
            self.downsample_factor = config['downsample_factor']
            self.noise_level = config['noise_level']

    # TODO(mchan): Add a preprocess flag to enable/disable the preprocessing steps.
    def load_real_data(self,
            show_im: bool = False) -> typing.Tuple[npt.NDArray, npt.NDArray]:
        """Loads PSF and diffuser cam image data.

        This function (1) reads a captured diffuser cam image and its corresponding calibrated point spread function (PSF) image, (2) downsamples those images, and (3) normalizes the intensity of those images.

        Args:
            show_im: Flag to enable/disable displaying the PSF and diffuser cam images.

        Returns:
            A tuple containing the PSF image and the diffuser cam image as 2D numpy arrays.

        Exceptions:
            IOError: Both input images must have the same dimensions.
        """
        if (self.measurement_fname is None):
            raise IOError('The path to the measurement image file was not specified in your configuration file.')
        psf = cv2.imread(self.psf_fname, cv2.IMREAD_GRAYSCALE).astype(float)
        data = cv2.imread(self.measurement_fname, cv2.IMREAD_GRAYSCALE).astype(float)
        
        if (psf.shape != data.shape):
            raise IOError('PSF image and diffuser cam image must be the same dimensions.')
        
        """In the picamera, there is a non-trivial background 
        (even in the dark) that must be subtracted"""
        bg = np.mean(psf[5:15,5:15]) 
        psf -= bg
        data -= bg
        
        psf = utils.downsample(psf, self.downsample_factor)
        data = utils.downsample(data, self.downsample_factor)
        
        """Now we normalize the images so they have the same total power. Technically not a
        necessary step, but the optimal hyperparameters are a function of the total power in 
        the PSF (among other things), so it makes sense to standardize it"""
        psf /= np.linalg.norm(psf.ravel())
        data /= np.linalg.norm(data.ravel())
        
        if show_im:
            plt.figure()
            plt.title("Point spread function")
            plt.imshow(psf)
            plt.figure()
            plt.title("Diffuser cam measurement")
            plt.imshow(data)
            plt.show()
        return psf, data

    def load_sim_data(
            self,
            show_im: bool = False) -> typing.Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """Generates a PSF and a simulated diffuser cam image that can be used to test the reconstruction algorithm.

        Convolves the ground truth image by the PSF to get an artificial diffuser cam measurement.

        Args:
            show_im: Flag to enable/disable displaying the simulated diffuser cam measurement image.

        Returns:
            A tuple containing the PSF image, the diffuser cam image, and the ground truth image as 2D numpy arrays.

        Exceptions:
            IOError: Both input images must have the same dimensions.
        """
        if (self.gt_fname is None):
            raise IOError('The path to the ground truth image file was not specified in your configuration file.')
        psf = cv2.imread(self.psf_fname, cv2.IMREAD_GRAYSCALE).astype(float)
        gt = cv2.imread(self.gt_fname, cv2.IMREAD_GRAYSCALE).astype(float)

        if (psf.shape != gt.shape):
            # TODO(mchan): Use the Python logger to print warnings.
            print('Resizing the PSF and ground truth images to be the same dimensions.')
            gt = cv2.resize(gt, psf.shape[::-1])


        psf = utils.downsample(psf, self.downsample_factor)
        gt = utils.downsample(gt, self.downsample_factor)
        psf /= np.linalg.norm(psf.ravel())
        gt /= np.linalg.norm(gt.ravel())

        # Convolves the ground truth image by the PSF to get a simulated blurry diffuser cam measurement.
        A = np.fft.fft2(np.fft.ifftshift(utils.pad(psf)), norm='ortho')
        b = np.fft.fft2(np.fft.ifftshift(utils.pad(gt)))
        data = np.real(utils.crop(np.fft.fftshift(np.fft.ifft2(A * b)), psf.shape))

        std_dev = np.amax(data) * self.noise_level
        noise = np.random.normal(0, std_dev, data.shape)
        data += noise

        if (show_im):
            plt.figure()
            plt.title("Simulated diffuser cam measurement")
            plt.imshow(data)
            plt.show()
        return psf, data, gt