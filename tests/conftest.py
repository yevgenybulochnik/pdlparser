import pytest
from pathlib import Path


@pytest.fixture
def pdf_sample_path():
    return (
        Path.cwd()
        / "tests"
        / "data"
        / "sample_pdf.pdf"
    ).as_posix()
