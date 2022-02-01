import numpy as np
import matplotlib.pyplot as plt

from . import utils

# TODO(mchan): Refactor matrix precomputation into a separate function.
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
    # Initializes matrices.
    # TODO(mchan): Better understand why the 1/sqrt(N) scaling factor matters here.
    A = np.fft.fft2(np.fft.ifftshift(utils.pad(psf)), norm='ortho')
    b = np.fft.fft2(np.fft.ifftshift(utils.pad(data)))
    x = np.ones(data.shape) * 0.5

    # The Hermitian (aka conjugate transpose) of a matrix A in the spatial domain is equal to the conjugate of DFT(A).
    Ah = A.conj()

    # Precomputes matrices outside of the iterative optimization loop.
    AhA = Ah * A
    Ahb = utils.crop(np.fft.fftshift(np.fft.ifft2(Ah * b)), data.shape)
    alpha = np.real(1.8/(np.max(AhA)))

    while n_iter > 0:
        xt = np.fft.fft2(np.fft.ifftshift(utils.pad(x)))
        gradient = utils.crop(np.fft.fftshift(np.fft.ifft2(AhA * xt)), data.shape) - Ahb
        x -= alpha * np.real(gradient)
        n_iter -= 1

    result = np.maximum(x, 0)
    if (show_im):
        plt.figure()
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
    # Initializes matrices.
    A = np.fft.fft2(np.fft.ifftshift(utils.pad(psf)), norm='ortho')
    b = np.fft.fft2(np.fft.ifftshift(utils.pad(data)))
    x = np.ones(data.shape) * 0.5

    # The Hermitian (aka conjugate transpose) of a matrix A in the spatial domain is equal to the conjugate of DFT(A).
    Ah = A.conj()

    # Precomputes matrices outside of the iterative optimization loop.
    AhA = Ah * A
    Ahb = Ah * b 

    while n_iter > 0:
        denom = AhA + alpha
        numerator = Ahb
        x += np.real(utils.crop(np.fft.fftshift(np.fft.ifft2(numerator / denom)), data.shape))
        n_iter -= 1

    result = np.maximum(x, 0)
    if (show_im):
        plt.figure()
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
    # Initializes matrices.
    A = np.fft.fft2(np.fft.ifftshift(utils.pad(psf)), norm='ortho')
    b = np.fft.fft2(np.fft.ifftshift(utils.pad(data)))
    x = np.ones(data.shape) * 0.5
    z = x.copy()
    u = np.zeros(x.shape)

    # The Hermitian (aka conjugate transpose) of a matrix A in the spatial domain is equal to the conjugate of DFT(A).
    Ah = A.conj()

    # Precomputes matrices outside of the iterative optimization loop.
    AhA = Ah * A
    Ahb = Ah * b

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
        plt.title("Reconstructed image")
        plt.imshow(result)
        plt.show()
    return result

# TODO(mchan): Add the UCB TV-ADMM implementation here.
def tv_admm():
    pass