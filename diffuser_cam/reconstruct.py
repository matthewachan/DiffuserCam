import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

from . import utils
from DnCNN.models import DnCNN
from DnCNN.utils import *

def sgd(psf, data, n_iter=3, show_im=False):
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

class Denoiser():
    def __init__(self, cuda=False):
        model_path = './DnCNN/logs/DnCNN-S-15/net.pth'
        n_layers = 17
        net = DnCNN(channels=1, num_of_layers=n_layers)
        device_ids = [0]

        self.cuda = cuda
        if not self.cuda:
            self.model = torch.nn.DataParallel(net, device_ids=device_ids)
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        else:
            self.model = torch.nn.DataParallel(net, device_ids=device_ids).cuda()
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def denoise(self, img):
        img = np.float32(img[:, :, 0])
        img = np.expand_dims(img, 0)
        img = np.expand_dims(img, 1)
        noisy = Variable()
        if self.cuda:
            noisy = Variable(torch.Tensor(img).cuda())
        else:
            noisy = Variable(torch.Tensor(img))
        with torch.no_grad():
            result = torch.clamp(noisy - self.model(noisy), 0., 1.)
        result = torch.squeeze(result.data.cpu(), 0).permute(1, 2, 0)
        return result.numpy()

def pnp_admm(psf, data, n_iter, rho, show_im=False):
    dncnn = Denoiser()
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