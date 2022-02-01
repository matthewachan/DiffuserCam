"""
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from . import utils

def load_data(psf, data, show_im=False, downsample_factor=1.0/8.0):
    """Loads PSF and diffuser cam image data.

    This function reads a captured diffuser cam image and its corresponding calibrated point spread function (PSF) image.

    Args:
        psf: Path to the PSF image or the image itself.
        data: Path to the diffuser cam image or the image itself.
        show_im: Flag to enable/disable displaying the PSF and diffuser cam images.
        downsample_factor: Factor to rescale the PSF and diffuser cam images by for faster processing speed.

    Returns:
        A tuple containing the PSF image and the diffuser cam image as 2D numpy arrays.

    Exceptions:
        IOError: Both input images must have the same dimensions.
    """
    # Checks if the input PSF and diffuser cam image are file paths.
    if (type(psf) == str):
        psf = cv2.imread(psf, cv2.IMREAD_GRAYSCALE).astype(float)
    if (type(data) == str):
        data = cv2.imread(data, cv2.IMREAD_GRAYSCALE).astype(float)
    
    if (psf.shape != data.shape):
        raise IOError("PSF image and diffuser cam image must be the same dimensions.")
    
    """In the picamera, there is a non-trivial background 
    (even in the dark) that must be subtracted"""
    bg = np.mean(psf[5:15,5:15]) 
    psf -= bg
    data -= bg
    
    psf = utils.downsample(psf, downsample_factor)
    data = utils.downsample(data, downsample_factor)
    
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

def load_sim_data(gt, psf, downsample_factor, show_im=False):
    """Generates a PSF and a simulated diffuser cam image that can be used to test the reconstruction algorithm.

    Args:
        gt: Path to the ground truth clear image or the image itself
        psf: Path to the PSF or the image itself
        downsample_factor: Factor to rescale the PSF and ground truth images by for faster processing speed.
        show_im: Flag to enable/disable displaying the simulated diffuser cam measurement image.

    Returns:
        A tuple containing the PSF image and the diffuser cam image as 2D numpy arrays.

    Exceptions:
        IOError: Both input images must have the same dimensions.
    """
    if (type(psf) == str):
        psf = cv2.imread(psf, cv2.IMREAD_GRAYSCALE).astype(float)
    if (type(gt) == str):
        gt = cv2.imread(gt, cv2.IMREAD_GRAYSCALE).astype(float)

    if (psf.shape != gt.shape):
        raise IOError("PSF image and diffuser cam image must be the same dimensions.")

    psf = utils.downsample(psf, downsample_factor)
    gt = utils.downsample(gt, downsample_factor)
    psf /= np.linalg.norm(psf.ravel())
    gt /= np.linalg.norm(gt.ravel())

    # Convolves the ground truth image by the PSF to get a simulated blurry diffuser cam measurement.
    A = np.fft.fft2(np.fft.ifftshift(utils.pad(psf)), norm='ortho')
    b = np.fft.fft2(np.fft.ifftshift(utils.pad(gt)))
    data = np.real(utils.crop(np.fft.fftshift(np.fft.ifft2(A * b)), psf.shape))

    # TODO(mchan): Add AWGN to the image.

    if (show_im):
        plt.figure()
        plt.title("Simulated diffuser cam measurement")
        plt.imshow(data)
        plt.show()
    return psf, data