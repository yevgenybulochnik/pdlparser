from cv2 import cv2
import numpy as np
from pdlparser.extract import extract_set

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


def generate_contours(img: Image):
    """Given a PIL image object generate contour lines.

    Function will generate contours based on vertical and horizontal table
    lines.

    Args:
        img: PIL image object

    """
    img_vh = generate_vh_lines(img)

    contours, _ = cv2.findContours(
        img_vh,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    return contours


def parse_contours(contours):
    """Parse generated contours.

    Function will parse generated contours into larger outer and inner
    rectangles.

    Args:
        contours: array

    """
    outer_rects = []
    inner_rects = []

    for cnt in contours:
        rect = cv2.boundingRect(cnt)
        _, _, width, _ = rect

        if width > 3300:
            continue
        if width > 3000:
            outer_rects.append(rect)
        else:
            inner_rects.append(rect)

    return outer_rects[::-1], inner_rects[::-1]


def parse_class_contours(outer_rects, inner_rects):
    """Parse contours into a class grouping.

    This functions takes outer and inner rectangles and parses them into a
    class datastructure to facilitate further parsing.

    Args:
        outer_rects: array
        inner_rects: array

    Returns:
        Array of class group dictionaries
    """

    class_groups = []

    for outer_rect in outer_rects:
        class_group = {
            'left': [],
            'right': [],
            'header': None,
            'content_box': None,
        }

        x, y, width, height = outer_rect

        for inner_rect in inner_rects:
            inner_x, inner_y, _, _ = inner_rect

            if y < inner_y < y + height:
                class_group['content_box'] = outer_rect
                header_rect = (
                    x,
                    y - 100,
                    width,
                    100
                )
                class_group['header'] = header_rect

                if inner_x < 1500:
                    class_group['left'].append(inner_rect)
                else:
                    class_group['right'].append(inner_rect)

        class_groups.append(class_group)

    return class_groups


def get_data(img: Image):
    """Given an image get all data from table structures.

    This functions combines multiple methods to parse an image, determine its
    table structure and extract all data from said tables.

    Args:
        img: PIL image object

    """
    contours = generate_contours(img)
    outer_rect, inner_rect = parse_contours(contours)
    class_groups = parse_class_contours(outer_rect, inner_rect)

    data = []

    for class_group in class_groups:
        class_data = extract_set(np.array(img), class_group)

        data.append(class_data)

    return data


def show_class_group(class_group, img_array):
    """Given a class_group dictionary display outlines on img_array.

    Args:
        class_group: class_group dictionary
        img_array: numpy array
    Returns:
        Returns image array
    """

    x, y, w, h = class_group['content_box']
    cv2.rectangle(img_array, (x, y), (x+w, y+h), (255, 0, 0), 10)

    x, y, w, h = class_group['header']
    cv2.rectangle(img_array, (x, y), (x+w, y+h), (0, 0, 255), 10)

    for left_rect in class_group['left']:
        x, y, w, h = left_rect
        cv2.rectangle(img_array, (x, y), (x+w, y+h), (0, 255, 0), 10)

    for right_rect in class_group['right']:
        x, y, w, h = right_rect
        cv2.rectangle(img_array, (x, y), (x+w, y+h), (165, 42, 42), 10)

    return img_array


def show_class_groups(class_groups, img):
    """Given an array of class_group dictionaries display outlines on an
    img_array

    Args:
        class_groups: array of class groups
        img: PIL image object

    Returns:
        Returns image numpy array

    """
    img_array = np.array(img)

    for class_group in class_groups:
        show_class_group(class_group, img_array)

    return img_array
