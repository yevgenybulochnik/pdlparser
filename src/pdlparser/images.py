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
