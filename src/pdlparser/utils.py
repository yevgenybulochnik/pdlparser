import tempfile
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
from pathlib import Path
from pdlparser.images import get_data

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
