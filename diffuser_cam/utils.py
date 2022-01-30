import numpy as np

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
    if factor >= 1 or factor < 0:
        raise IOError(
            "The image downsampling factor must be between 0 and 1 (non inclusive).")

    num = int(-np.log2(factor))
    for i in range(num):
        img = 0.25*(img[::2, ::2, ...]+img[1::2, ::2, ...] +
                    img[::2, 1::2, ...]+img[1::2, 1::2, ...])
    return img
