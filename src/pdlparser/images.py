from cv2 import cv2
import numpy as np

import numpy.typing as npt
from PIL.Image import Image


def generate_img_bin(img: Image) -> npt.NDArray:
    """Generate a reverse grayscale image returned as a numpy array.

    Args:
        img: PIL image object

    Returns:
        Returns a numpy array based off the PIL image.
    """
    img_array = np.array(img)
    _, img_bin = cv2.threshold(img_array, 128, 255, cv2.THRESH_BINARY)

    img_reversal = 255 - img_bin

    return img_reversal


def generate_kernels(img_array: npt.NDArray):
    """Given an binary image array generate a standard kernel, vertical kernal
    and horizontal kernal.

    Args:
        img_array: numpy array (binary_image)
    Returns:
        Tuple including kernal, vertical kernal and horizontal kernal
    """
    kernel_len = img_array.shape[1] // 135

    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))

    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    return kernel, vert_kernel, hori_kernel
