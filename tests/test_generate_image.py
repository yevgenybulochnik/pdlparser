from pathlib import Path
from pdlparser.utils import generate_image


def test_generate_image_page_param(pdf_sample_path):
    """
    Given the page param as an int or tuple slice the pdf and return an image
    """
    img = generate_image(pdf_sample_path)
    img = generate_image(pdf_sample_path, page=1)
    img = generate_image(pdf_sample_path, page=(1, 3))

    assert img


def test_generate_image_pdf_path_param(pdf_sample_path):
    """
    Given str path or pathlib object generate image from PDF
    """
    pathlib_object = Path(pdf_sample_path)
    img = generate_image(pdf_sample_path)
    img = generate_image(pathlib_object)
    assert img
