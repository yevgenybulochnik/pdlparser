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


def generate_vh_lines(img: Image) -> npt.NDArray:
    """Given a PIL image object create a new image with just table outline
    elements.

    This function takes an image and uses errosion and dilation to bring out
    vertical and horizontal lines of tables.

    Args:
        img: PIL image object

    Returns:
        Numpy array with just table elements.

    """
    img_bin = generate_img_bin(img)
    kernel, vert_kernel, hori_kernel = generate_kernels(img_bin)

    v1 = cv2.erode(img_bin, vert_kernel, iterations=3,)
    vert_lines = cv2.dilate(v1, vert_kernel, iterations=3,)

    h1 = cv2.erode(img_bin, hori_kernel, iterations=3,)
    hori_lines = cv2.dilate(h1, hori_kernel, iterations=3,)

    img_vh = cv2.addWeighted(vert_lines, 0.5, hori_lines, 0.5, 0)
    img_vh = cv2.erode(~img_vh, kernel, iterations=2)

    _, img_vh = cv2.threshold(img_vh, 128, 255, cv2.THRESH_BINARY)

    img_vh = cv2.cvtColor(img_vh, cv2.COLOR_BGR2GRAY)

    return img_vh
