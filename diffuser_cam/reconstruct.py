from . import utils
import numpy as np

def sgd(psf, data, n_iter=3):
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
    return np.maximum(x, 0)

# TODO(mchan): Move previous PnP-ADMM implementation here.
def pnp_admm(psf, data, n_iter, rho):
    pass

# TODO(mchan): Add the UCB TV-ADMM implementation here.
def tv_admm():
    pass