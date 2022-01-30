"""
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from . import utils

def load_data(psf_fname, data_fname, show_im=False, downsample_factor=1.0/8.0):
    """Loads PSF and diffuser cam image data.

    This function reads a captured diffuser cam image and its corresponding calibrated point spread function (PSF) image.

    Args:
        psf_fname: Path to the PSF image.
        data_fname: Path to the diffuser cam image.
        show_im: Flag to enable/disable displaying the PSF and diffuser cam images.
        downsample_factor: Factor to rescale the PSF and diffuser cam images by for faster processing speed.

    Returns:
        A tuple containing the PSF image and the diffuser cam image as 2D numpy arrays.
    """
    psf = cv2.imread(psf_fname, cv2.IMREAD_GRAYSCALE).astype(float)
    data = cv2.imread(data_fname, cv2.IMREAD_GRAYSCALE).astype(float)
    
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
        plt.imshow(psf)
        plt.figure()
        plt.imshow(data)
        plt.show()
    return psf, data