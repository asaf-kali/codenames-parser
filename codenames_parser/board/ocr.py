import logging
import os.path
from dataclasses import dataclass

import platformdirs
import requests

from codenames_parser.common.utils.models import Box

DEFAULT_TESSERACT_FOLDER = "/usr/share/tesseract-ocr/4.00/tessdata"
TESSDATA_REPO = "https://github.com/tesseract-ocr/tessdata"

log = logging.getLogger(__name__)


@dataclass
class WordIndex:
    page: int
    block: int
    paragraph: int
    line: int


@dataclass
class TesseractResult:
    text: str
    confidence: float
    box: Box
    level: int
    index: WordIndex

    @property
    def is_word(self) -> bool:
        return self.level == 5


class TesseractLanguageNotAvailable(Exception):
    pass


def fetch_tesseract_language(language: str):
    data_folder = os.getenv("TESSDATA_PREFIX", default=DEFAULT_TESSERACT_FOLDER)
    default_data_file = _get_tesseract_data_file_path(data_folder=data_folder, language=language)
    if os.path.exists(default_data_file):
        log.debug(f"Language data for '{language}' already exists in default location")
        return
    try:
        user_data_folder = _get_user_data_folder()
        _fetch_tesseract_language(data_folder=user_data_folder, language=language)
        return
    except Exception as e:
        log.warning(f"Could not fetch Tesseract language data for '{language}': {e}")
    temp_folder = os.path.join("/tmp", "tesseract")
    _fetch_tesseract_language(data_folder=temp_folder, language=language)


def _fetch_tesseract_language(data_folder: str, language: str):
    os.makedirs(data_folder, exist_ok=True)
    if not os.path.exists(data_folder):
        raise TesseractLanguageNotAvailable(f"Could not create data folder '{data_folder}'")
    os.environ["TESSDATA_PREFIX"] = data_folder
    data_file = f"{data_folder}/{language}.traineddata"
    if os.path.exists(data_file):
        log.debug(f"Language data for '{language}' already exists")
        return
    remote_file = f"{TESSDATA_REPO}/raw/main/{language}.traineddata"
    log.info(f"Downloading '{language}' language data for Tesseract from '{remote_file}'")
    try:
        _download_file(source=remote_file, destination=data_file)
    except Exception as e:
        raise TesseractLanguageNotAvailable(f"Could not download language data for '{language}'") from e
    log.info(f"Downloaded '{language}' language data for Tesseract")


def _get_tesseract_data_file_path(data_folder: str, language: str):
    return os.path.join(data_folder, f"{language}.traineddata")


def _get_user_data_folder():
    return platformdirs.user_data_dir("tesseract")


def _download_file(source: str, destination: str):
    get_request = requests.get(source, timeout=30)
    get_request.raise_for_status()
    with open(destination, "wb") as f:
        f.write(get_request.content)


def parse_tesseract_data(data: dict) -> list[TesseractResult]:
    results = []
    n_boxes = len(data["text"])
    for i in range(n_boxes):
        result = _parse_result(data, i)
        results.append(result)
    return results


def _parse_result(data: dict, i: int) -> TesseractResult:
    text = data["text"][i]
    confidence = int(data["conf"][i])
    x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
    box = Box(x=x, y=y, w=w, h=h)
    level = int(data["level"][i])
    index = WordIndex(
        page=int(data["page_num"][i]),
        block=int(data["block_num"][i]),
        paragraph=int(data["par_num"][i]),
        line=int(data["line_num"][i]),
    )
    result = TesseractResult(text=text, confidence=confidence, box=box, level=level, index=index)
    return result
