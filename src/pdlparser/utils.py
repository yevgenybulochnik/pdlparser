import numpy as np
from pdf2image import convert_from_path
from PIL import Image
from pathlib import Path

from typing import Union


def generate_image(
    pdf_path: Union[str, Path],
    dpi: int = 150,
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
