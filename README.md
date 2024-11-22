# Codenames Parser

[![PyPI version](https://badge.fury.io/py/codenames-parser.svg)](https://badge.fury.io/py/codenames-parser)
[![Pipeline](https://github.com/asaf-kali/codenames-parser/actions/workflows/pipeline.yml/badge.svg)](https://github.com/asaf-kali/codenames-parser/actions/workflows/pipeline.yml)
[![codecov](https://codecov.io/github/asaf-kali/codenames-parser/graph/badge.svg?token=HET5E8P1UK)](https://codecov.io/github/asaf-kali/codenames-parser)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-111111.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/imports-isort-%231674b1)](https://pycqa.github.io/isort/)
[![Type checked: mypy](https://img.shields.io/badge/type%20check-mypy-22aa11)](http://mypy-lang.org/)
[![Linting: pylint](https://img.shields.io/badge/linting-pylint-22aa11)](https://github.com/pylint-dev/pylint)

A Python package to parse Codenames game boards from images.\
Before we dive in, here are some examples:

### Color map extraction

Given `color_map.png`: \
<img src="./tests/fixtures/color_maps/classic/top_view.png" width="400"/> \
Running:

```
python -m codenames_parser/color_map/entrypoint.py color_map.png classic
```

Outputs:

```
Some parsing logs...

# As emoji table
⬜ 🟥 🟦 🟦 🟥
⬜ 🟦 🟥 ⬜ 🟦
🟦 🟥 🟥 🟦 ⬜
🟦 🟥 🟥 ⬜ ⬜
⬜ 🟦 💀 🟥 🟦

# As list
['NEUTRAL', 'RED', 'BLUE', 'BLUE', 'RED', 'NEUTRAL', 'BLUE', 'RED', 'NEUTRAL', 'BLUE', 'BLUE', 'RED', 'RED', 'BLUE', 'NEUTRAL', 'BLUE', 'RED', 'RED', 'NEUTRAL', 'NEUTRAL', 'NEUTRAL', 'BLUE', 'ASSASSIN', 'RED', 'BLUE']
```

### Board extraction

"Life is not perfect, neither is OCR" (credit: Github Copilot)

Given `board.png`: \
<img src="./tests/fixtures/boards/heb/board3_top.jpg" width="400"/> \
Running:

```
python -m codenames_parser/board/entrypoint.py board.png heb
```

Outputs:

```
Some parsing logs...

# As table
+-------+---------+-------+-------+------+
| ציבור | אוטובוס | ישראל |  מתח  |  גס  |
+-------+---------+-------+-------+------+
| ברית  |   גוש   | איום  | מורח  | קנה  |
+-------+---------+-------+-------+------+
| לידה  |  מבחן   | אודם  | שוקו  | חטיף |
+-------+---------+-------+-------+------+
|  חוק  |   רץ    | חצות  | רדיו  | כתם  |
+-------+---------+-------+-------+------+
|  גרם  |   כהן   | רושם  | אלמוג |      |
+-------+---------+-------+-------+------+

# As list
[
    "ציבור",    "אוטובוס",    "ישראל",    "מתח",    "גס",
    "ברית",    "גוש",    "איום",    "מורח",    "קנה",
    "לידה",    "מבחן",    "אודם",    "שוקו",    "חטיף",
    "חוק",    "רץ",    "חצות",    "רדיו",    "כתם",
    "גרם",    "כהן",    "רושם",    "אלמוג",     "",
]
```

\* It looks as if the direction of the board is flipped.
This is a bug due to Hebrew being written from right to left, the list order is correct.

## Installation

`pip install codenames-parser`

### OCR

1. Uses `pytesseract` to extract text from images.
2. Requires `tesseract` to be installed on the system (
   see [installing-tesseract](https://github.com/tesseract-ocr/tesseract/tree/main?tab=readme-ov-file#installing-tesseract)).
3. Download more languages from: https://github.com/tesseract-ocr/tessdata
