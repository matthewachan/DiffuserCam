"""
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import typing

from diffuser_cam import utils

class Algorithm:
    """Runs a reconstruction algorithm on a PSF and an image taken on a diffuser cam.

    The Algorithm configuration file should be in JSON format and can contain the following configuration parameters:
        algorithm: Name of the reconstruction algorithm to use (either "admm", "sgd", or "tikhonov").
        n_iter: Number of iterations to run the optimization loop for.
        rho: Floating point hyperparameter required by ADMM. 
        alpha: Floating point hyperparameter required by the Tikhonov reconstruction algorithm.

    Attributes:
        config: Dictionary of configuration parameters for the reconstruction algorithm.
    """
    def __init__(self, config_fname) -> None:
        """Loads the configuration file.
        
        Args:
            config_fname: Path to the JSON configuration file.
        """
        if (type(config_fname) == str):
            with open(config_fname) as f:
                self.config = json.load(f)
        else:
            self.config = config_fname

    def reconstruct(self, psf: npt.NDArray,
                    data: npt.NDArray,
                    show_im: bool = False) -> npt.NDArray:
        """Reconstructs the scene given a PSF and diffuser cam image.

        Args:
            psf: The point spread function image.
            data: The blurry diffuser cam image.
            show_im: Flag to enable/disable displaying the final reconstructed image.

        Returns:
            The reconstructed image.
        """
        algorithm_name = self.config['algorithm']
        n_iter = self.config['n_iter']

        if (algorithm_name == 'admm'):
            return pnp_admm(psf, data, n_iter, self.config['rho'], show_im)
        elif (algorithm_name == 'tikhonov'):
            return tikhonov(psf, data, n_iter, self.config['alpha'], show_im)
        elif (algorithm_name == 'sgd'):
            return sgd(psf, data, n_iter, show_im)
        else:
            raise IOError('The reconstruction algorithm you specified is invalid.')

def sgd(psf, data, n_iter, show_im=False):
    """Reconstructs the scene using stochastic gradient descent.

    Args:
        psf: The point spread function image.
        data: The blurry diffuser cam image.
        n_iter: Number of iterations of the optimization loop.
        show_im: Flag to enable/disable displaying the final reconstructed image.

    Returns:
        The reconstructed image.
    """
    x, AhA, Ahb = utils.precompute_matrices(psf, data)
    Ahb = utils.crop(np.fft.fftshift(np.fft.ifft2(Ahb)), data.shape)
    alpha = np.real(1.8/(np.max(AhA)))

    while n_iter > 0:
        xt = np.fft.fft2(np.fft.ifftshift(utils.pad(x)))
        gradient = utils.crop(np.fft.fftshift(np.fft.ifft2(AhA * xt)), data.shape) - Ahb
        x -= alpha * np.real(gradient)
        n_iter -= 1

    result = np.maximum(x, 0)
    if (show_im):
        plt.figure()
        plt.title('SGD reconstruction')
        plt.imshow(result)
        plt.show()
    return result

def tikhonov(psf, data, n_iter, alpha, show_im=False):
    """Reconstructs the scene using Tikhonov regularization (aka ridge regression).

    For the sake of simplicity, the Tikhonov matrix used is the identity matrix scaled by a parameter alpha.

    Args:
        psf: The point spread function image.
        data: The blurry diffuser cam image.
        n_iter: Number of iterations of the optimization loop.
        alpha: Scalar hyperparameter that is used to construct the Tikhonov matrix. In this implementation, the Tikhonov matrix is simply the identity matrix scaled by this parameter, alpha.
        show_im: Flag to enable/disable displaying the final reconstructed image.

    Returns:
        The reconstructed image.
    """
    x, AhA, Ahb = utils.precompute_matrices(psf, data)

    while n_iter > 0:
        denom = AhA + alpha
        numerator = Ahb
        x += np.real(utils.crop(np.fft.fftshift(np.fft.ifft2(numerator / denom)), data.shape))
        n_iter -= 1

    result = np.maximum(x, 0)
    if (show_im):
        plt.figure()
        plt.title('Tikhonov reconstruction')
        plt.imshow(result)
        plt.show()
    return result

def pnp_admm(psf, data, n_iter, rho, show_im=False, cuda=False):
    """Reconstructs the scene using plug-and-play ADMM.

    Args:
        psf: The point spread function image.
        data: The blurry diffuser cam image.
        n_iter: Number of iterations of the optimization loop.
        rho: Scalar Lagrange multiplier.
        show_im: Flag to enable/disable displaying the final reconstructed image.
        cuda: Flag to enable/disable running the denoiser model on GPU.

    Returns:
        The reconstructed image.
    """
    dncnn = utils.Denoiser(cuda)
    x, AhA, Ahb = utils.precompute_matrices(psf, data)
    z = x.copy()
    u = np.zeros(x.shape)

    while n_iter > 0:
        # Updates the proximal operator for x.
        z_fft = np.fft.fft2(np.fft.ifftshift(utils.pad(z)))
        u_fft = np.fft.fft2(np.fft.ifftshift(utils.pad(u)))
        numerator = Ahb + rho * (z_fft - u_fft) 
        denom = AhA + rho
        x = np.real(utils.crop(np.fft.fftshift(np.fft.ifft2(numerator / denom)), data.shape))

        # Updates the proximal operator for z.
        z = dncnn.denoise(np.expand_dims(x + u, axis=2))
        z = np.squeeze(z)

        # Updates the proximal operator for u.
        u += (x - z)

        n_iter -= 1

    result = np.maximum(x, 0)
    if (show_im):
        plt.figure()
        plt.title('ADMM reconstruction')
        plt.imshow(result)
        plt.show()
    return result

# TODO(mchan): Add the UCB TV-ADMM implementation here.
def tv_admm():
    pass