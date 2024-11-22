# Codenames Parser

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

### OCR

1. Uses `pytesseract` to extract text from images.
2. Requires `tesseract` to be installed on the system (
   see [installing-tesseract](https://github.com/tesseract-ocr/tesseract/tree/main?tab=readme-ov-file#installing-tesseract)).
3. Download more languages from: https://github.com/tesseract-ocr/tessdata
