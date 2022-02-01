import numpy as np
import cv2
import torch
from torch.autograd import Variable

from DnCNN.models import DnCNN
from DnCNN.utils import *

def compute_padding(n):
    """ Returns the nearest power of 2 that is larger than 2N. 
    
    This optimization relies on the fact that
    the Discrete Fourier Transform leverages small prime factors for speed up.

    Args:
        n: The input dimension that we'd like to pad
    """
    return int(np.power(2, np.ceil(np.log2(2 * n - 1))))

def pad(input):
    """ Returns the input image padded so that its dimensions are the next power of 2 larger than 2w and 2h.

    Args:
        input: The image to pad
    """
    h, w = compute_padding(input.shape[0]), compute_padding(input.shape[1])
    v_pad = (h - input.shape[0]) // 2
    h_pad = (w - input.shape[1]) // 2
    padded = cv2.copyMakeBorder(
        input, v_pad, v_pad, h_pad, h_pad, cv2.BORDER_CONSTANT)
    return padded


def crop(input, original_shape):
    """ Returns the cropped version of a padded image

    Args:
        input: Padded image
        original_shape: Original dimensions of an image padded with pad()
    """
    diff = np.subtract(input.shape, original_shape)
    assert(np.all(diff >= 0))
    h, w = original_shape
    h_pad, w_pad = diff // 2
    return input[h_pad:(input.shape[0] - h_pad), w_pad:(input.shape[1] - w_pad)]


def downsample(img, factor):
    """Downsamples an image.

    Downsamples an image to smaller size and smoothes the result with a box filter to prevent aliasing artifacts.

    Args:
        img: The image to downsample.
        factor: The amount to downsample by

    Returns:
        The downsampled image.

    Raises:
        IOError: Input downsample factor is outside of (0, 1).

    """
    if factor > 1 or factor < 0:
        raise IOError(
            "The image downsampling factor must be between 0 (exclusive) and 1 (inclusive).")
    if factor == 1:
        return img

    num = int(-np.log2(factor))
    for i in range(num):
        img = 0.25*(img[::2, ::2, ...]+img[1::2, ::2, ...] +
                    img[::2, 1::2, ...]+img[1::2, 1::2, ...])
    return img

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