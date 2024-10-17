import logging
import os.path

import platformdirs
import requests

DEFAULT_TESSERACT_FOLDER = "/usr/share/tesseract-ocr/4.00/tessdata"
TESSDATA_REPO = "https://github.com/tesseract-ocr/tessdata"

log = logging.getLogger(__name__)


class TesseractLanguageNotAvailable(Exception):
    pass


def fetch_tesseract_language(language: str):
    default_data_file = _get_tesseract_data_file_path(data_folder=DEFAULT_TESSERACT_FOLDER, language=language)
    if os.path.exists(default_data_file):
        log.info(f"Language data for '{language}' already exists in default location")
        return
    user_data_folder = _get_user_data_folder()
    _fetch_tesseract_language(data_folder=user_data_folder, language=language)


def _fetch_tesseract_language(data_folder: str, language: str):
    os.makedirs(data_folder, exist_ok=True)
    if not os.path.exists(data_folder):
        raise TesseractLanguageNotAvailable(f"Could not create data folder '{data_folder}'")
    os.environ["TESSDATA_PREFIX"] = data_folder
    data_file = f"{data_folder}/{language}.traineddata"
    if os.path.exists(data_file):
        log.info(f"Language data for '{language}' already exists")
        return
    remote_file = f"{TESSDATA_REPO}/raw/main/{language}.traineddata"
    log.info(f"Downloading '{language}' language data for Tesseract from '{remote_file}'")
    try:
        _download_file(source=remote_file, destination=data_file)
    except Exception as e:
        raise TesseractLanguageNotAvailable(f"Could not download language data for '{language}'") from e
    log.info(f"Downloaded '{language}' language data for Tesseract")


def _get_tesseract_data_file_path(data_folder: str, language: str):
    return f"{data_folder}/{language}.traineddata"


def _get_user_data_folder():
    return platformdirs.user_data_dir("tesseract")


def _download_file(source: str, destination: str):
    get_request = requests.get(source, timeout=30)
    get_request.raise_for_status()
    with open(destination, "wb") as f:
        f.write(get_request.content)
