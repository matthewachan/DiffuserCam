import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import argparse
import glob
import torch
import torch.nn as nn
from torch.autograd import Variable
from DnCNN.models import DnCNN
from DnCNN.utils import *


""" TODO: Move these config variables into a separate config file. """
rho = 0.01
num_iter = 3
f = 1.0 / 8.0 # Downsample factor

def show_image(img):
    cv2.imshow("", img*255)
    #waits for user to press any key 
    #(this is necessary to avoid Python kernel form crashing)
    cv2.waitKey(0) 
    #closing all open windows 
    cv2.destroyAllWindows() 

""" Adjusted from UC Berkeley's code """
def loadData(psf_fname, data_fname, show_im=True):
    psf = cv2.imread(psf_fname, cv2.IMREAD_GRAYSCALE).astype(float)
    data = cv2.imread(data_fname, cv2.IMREAD_GRAYSCALE).astype(float)
    print("Max RGB after reading: ", np.max(psf), np.max(data))
    
    """In the picamera, there is a non-trivial background 
    (even in the dark) that must be subtracted"""
    bg = np.mean(psf[5:15,5:15]) 
    psf -= bg
    data -= bg
    print("Max RGB after subtracting background: ", np.max(psf), np.max(data))
    
    """Resize to a more manageable size to do reconstruction on. 
    Because resizing is downsampling, it is subject to aliasing 
    (artifacts produced by the periodic nature of sampling). Demosaicing is an attempt
    to account for/reduce the aliasing caused. In this application, we do the simplest
    possible demosaicing algorithm: smoothing/blurring the image with a box filter"""
    
    def resize(img, factor):
        num = int(-np.log2(factor))
        for i in range(num):
            img = 0.25*(img[::2,::2,...]+img[1::2,::2,...]+img[::2,1::2,...]+img[1::2,1::2,...])
        return img
    
    f = 0.25
    psf = resize(psf, f)
    data = resize(data, f)
    
    """Now we normalize the images so they have the same total power. Technically not a
    necessary step, but the optimal hyperparameters are a function of the total power in 
    the PSF (among other things), so it makes sense to standardize it"""
    
    psf /= np.linalg.norm(psf.ravel())
    data /= np.linalg.norm(data.ravel())
    print("Max RGB after norm: ", np.max(psf), np.max(data))
    
    if show_im:
        show_image(psf)
        show_image(data)
    return psf, data

""" Returns the nearest power of 2 that is larger than 2N. This optimization relies on the fact that
the Discrete Fourier Transform leverages small prime factors for speed up.
" n is the input dimension that we'd like to pad
"""
def compute_padding(n):
    return int(np.power(2, np.ceil(np.log2(2 * n - 1))))

test_n = 3
assert compute_padding(test_n) == 8, 'The padding calcuations are incorrect.'

""" Returns the input image padded so that its dimensions are the next power of 2 larger than 2w and 2h.
" input is the image to pad
"""
def pad(input):
    h, w = compute_padding(input.shape[0]), compute_padding(input.shape[1])
    padded = cv2.copyMakeBorder(input, h // 2, h // 2, w // 2, w // 2, cv2.BORDER_CONSTANT)
    return padded

test_img = cv2.imread('./img/psf.tif', cv2.IMREAD_GRAYSCALE)
assert pad(test_img).shape == (5296, 5696), 'The padded dimensions are incorrect.'

""" Returns the cropped version of a padded image
" input is the padded image
" original_shape is the original dimensions of an image padded with pad() """
def crop(input, original_shape):
    diff = np.subtract(input.shape, original_shape)
    assert(np.all(diff >= 0))
    h, w = original_shape
    h_pad, w_pad = diff // 2
    return input[h_pad:(input.shape[0] - h_pad), w_pad:(input.shape[1] - w_pad)]

assert crop(pad(test_img), test_img.shape).shape == test_img.shape, 'Cropped dimensions are incorrect.'

""" Resizes the input image
" img is the image to resize
" factor is the resizing factor
"""
def resize(img, factor):
    num = int(-np.log2(factor))
    for i in range(num):
        img = 0.25 * (img[::2,::2,...] + img[1::2,::2,...] + img[::2,1::2,...] + img[1::2,1::2,...])
    return img

class Denoiser():
    def __init__(self):
        n_layers = 17
        net = DnCNN(channels=1, num_of_layers=n_layers)

        device_ids = [0]
        self.model = nn.DataParallel(net, device_ids=device_ids).cuda()
        model_path = './DnCNN/logs/DnCNN-S-15/net.pth'
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def denoise(self, img):
        img = np.float32(img[:, :, 0])
        img = np.expand_dims(img, 0)
        img = np.expand_dims(img, 1)
        noisy = Variable(torch.Tensor(img).cuda())
        with torch.no_grad():
            result = torch.clamp(noisy - self.model(noisy), 0., 1.)
        result = torch.squeeze(result.data.cpu(), 0).permute(1, 2, 0)
        return result.numpy()

# Run this to visualize the denoiser's performance.
# dncnn = Denoiser()
# noisy = cv2.imread('noisy.jpg')
# dncnn.denoise(noisy)
""" Solve argminx Ax = b.
" A is the 2D PSF image
" b is the measurement
"""
def pnp_admm(b, A, rho=1, num_iter=5, debug=False):
    orig_dims = A.shape
    dncnn = Denoiser()
    # Sets the initial state of optimization variables x, z, u.
    x = pad(np.ones(b.shape) * 0.5)
    z = x.copy()
    u = np.zeros(x.shape)

    # Precomputes matrix math.
    AtA = np.abs(np.fft.fft2(pad(A))) ** 2
    Atb = cv2.filter2D(pad(b), -1, A, borderType=cv2.BORDER_WRAP)

    for i in np.arange(num_iter):
        # Proximal update for x := x^{t+1} = (AtA + rho)^{-1} (Atb + rho * z - lambda)
        num = np.fft.fft2(Atb + rho * (z - u))
        denom = (AtA + rho)
        x = np.real((np.fft.ifft2(num / denom)))
        # Proximal update for z := D(x+u)
        z = dncnn.denoise(np.expand_dims(x + u, axis=2))
        z = np.squeeze(z)
        # Proximal update for u.
        u += (x - z)
        if debug:
            plt.figure()
            plt.title(f'PnP ADMM iteration {i}')
            plt.imshow(crop(np.clip(x, 0, 1) * 255, b.shape), cmap='gray')
            plt.show()
    # Clamps the RGB image between 0 and 1.
    x = np.clip(x, 0, 1)
    return crop(x, b.shape)

""" Utility function to threshold the background in the PSF image. This is helpful because the PSF measurements in our lab setup are noisy.
" psf is the PSF image to filter
"""
def filter_psf(psf, threshold=0.3):
    psf[psf < (threshold * 255)] = 0
    return psf

""" Reconstructs the scene.
" psf is the filename of the PSF iamge
" imgs is an array of DiffuserCam images
" downsample_factor is the amount to downsample the PSF and DiffuserCam images
" filter determines whether to filter the input PSF image
" show_plots determines whether image plots will be displayed
"""
def reconstruct(rho=1, num_iter=3, filter=True, show_plots=False, rgb=True):
    # Read in the PSF image.
    # psf = cv2.imread(psf, cv2.IMREAD_GRAYSCALE).astype(float)
    # if filter:
    #     psf = filter_psf(psf)
    # psf = resize(psf, downsample_factor) / 255

    # outputs = []

    return pnp_admm(data, psf, rho, num_iter)

    # Iterate through the images and reconstruct them one at a time.
    # for img in imgs:
    #     # img = cv2.imread(img).astype(float) if rgb else cv2.imread(img, cv2.IMREAD_GRAYSCALE).astype(float)
    #     img = resize(img, downsample_factor) / 255
    #     if rgb:
    #         r = pnp_admm(img[:,:,0], psf, rho, num_iter)
    #         g = pnp_admm(img[:,:,1], psf, rho, num_iter)
    #         b = pnp_admm(img[:,:,2], psf, rho, num_iter)
    #         reconstruction = np.dstack((r, g, b))
    #     else:
    #         reconstruction = pnp_admm(img, psf, rho, num_iter)
    #     # reconstruction *= 255
    #     outputs.append(reconstruction)
    #     if show_plots:
    #         plt.figure(figsize=(30, 12))
    #         plt.subplot(1, 3, 1)
    #         plt.title('PSF')
    #         plt.imshow(psf, cmap='gray')
    #         plt.subplot(1, 3, 2)
    #         plt.title('DiffuserCam image')
    #         plt.imshow(img, cmap='gray') if not rgb else plt.imshow(img)
    #         plt.subplot(1, 3, 3)
    #         plt.title(f'Reconstruction (rho={rho}, n_iter={num_iter})')
    #         plt.imshow(reconstruction, cmap='gray') if not rgb else plt.imshow(reconstruction)
    #     # plt.show()
    # return outputs

# Loads image and PSF
psf, data = loadData('img/psf.tif', 'img/measurement.tif', show_im=True)
print("Shape of all inputs: ", psf.shape, data.shape)
result = pnp_admm(data, psf, 0.01, 10)
print("Shape after PnP-ADMM: ", psf.shape, data.shape, result.shape)
show_image(result)
# result = cv2.resize(result, (data.shape[1], data.shape[0]))

# Applies PSF kernel to blur the image---this generates synthetic DiffuserCam image.
# psf = resize(psf, downsample)
# x_tilde = cv2.filter2D(result, -1, psf, borderType=cv2.BORDER_WRAP)
# x_tilde = cv2.resize(x_tilde, (x.shape[1], x.shape[0]))
# show_image(x_tilde)

# TODO: Figure out the difference between these two. Might need to clamp?

# Applies added white Gaussian noise to the image.

# Runs PnP-ADMM to perform blind deconvolution. This tests the performance of the algorithm.
# print(psf.shape, x_tilde.shape)
# y_tilde = reconstruct(psf, [x_tilde], filter=False, show_plots=False, rgb=False)
# plt.figure()
# plt.title("Artificial reconstruction")
# plt.imshow(y_tilde[0])

# """ Artificially blurs a clear image using the PSF, then does blind deconvolution to retrieve the clear image back """
# def simulation(psf, img):
#     blur = cv2.filter2D(img, -1, psf, borderType=cv2.BORDER_WRAP)
#     print(blur.shape)
#     plt.figure()
#     plt.title("Synthetically blurred image")
#     plt.imshow(blur)
#     y = reconstruct(psf, [blur], filter=False, show_plots=False, rgb=False)
#     plt.figure()
#     plt.title("Reconstructed image")
#     plt.imshow(y[0])
#     plt.figure()
#     plt.title("Original image")
#     plt.imshow(img)
#     print(img.shape, y[0].shape)

# psf = cv2.imread('img/psf.tif', cv2.IMREAD_GRAYSCALE).astype(float)
# y = cv2.imread('img/test/USAF.jpg', cv2.IMREAD_GRAYSCALE).astype(float)
# y = cv2.resize(y, (psf.shape[1], psf.shape[0]))
# simulation(psf, y)