import tempfile
import pandas as pd
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
from pathlib import Path
from pdlparser.images import get_data
from cv2 import cv2

from typing import Union


def generate_image(
    pdf_path: Union[str, Path],
    dpi: int = 400,
    page: Union[int, tuple] = (None, None),
):
    """Generate a JPEG image given the str path of a pdf file

    Args:
        pdf: string or pathlib path
        dpi: int
        page: tuple
    """

    if isinstance(pdf_path, Path):
        pdf_path = pdf_path.as_posix()

    if isinstance(page, int):
        page = (page, page + 1)

    images = convert_from_path(pdf_path, dpi)

    image_array = np.concatenate(images[slice(*page)], axis=0)

    return Image.fromarray(image_array)


def parse_pdl(pdf_path: Union[str, Path]):
    """Given a pdl pdf parse the pdf and return the table data

    This function will take a PDF path, convert the pdf pages to JPEG images
    and run image processing to extract data in table elements. This function
    is extremely memory intensive.

    Args:
        pdf_path: string or pathlib object

    Returns:
        Returns an array of extracted data

    """
    if isinstance(pdf_path, Path):
        pdf_path = pdf_path.as_posix()

    staging_data = []

    with tempfile.TemporaryDirectory() as path:
        images = convert_from_path(pdf_path, dpi=400, output_folder=path)

        for image in images:
            image_data = get_data(image)

            staging_data += image_data

    final_data = []
    for datum in staging_data[2:]:
        if not datum['header']:
            class_group = final_data[-1]
            class_group['non-preferred'] += datum['non-preferred']
            class_group['preferred'] += datum['preferred']
            continue

        final_data.append(datum)

    return final_data


def generate_df(data):
    """Given parsed data from a PDF return a pandas dataframe

    This function takes the result of of parse_pdl and returns a pandas
    dataframe

    Args:
        data: Array of class_group objects

    Returns:
        Pandas dataframe
    """

    dfs = []

    for class_set in data:
        header = class_set['header'][0]
        preferred = class_set['preferred']
        non_preferred = class_set['non-preferred']

        preferred_df = pd.DataFrame({
            'class': header,
            'status': 'preferred',
            'drug_label': preferred,
        })

        non_preferred_df = pd.DataFrame({
            'class': class_set['header'][0],
            'status': 'non-preferred',
            'drug_label': non_preferred,
        })

        dfs.append(
            pd.concat([preferred_df, non_preferred_df])
        )

    return pd.concat(dfs)


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
