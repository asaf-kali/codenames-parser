from typing import NamedTuple


class ParseBoardTestCase(NamedTuple):
    fixture_file: str
    language: str
    expected_words: list[str]
    allowed_errors: int = 1


BOARD_1_WORDS = [
    "קוסם",
    "פסל",
    "דיונון",
    "עגיל",
    "קבוצה",
    "קרחון",
    "נמלה",
    "שקוף",
    "מכשף",
    "עורב",
    "נפח",
    "צמח",
    "ברזל",
    "מערה",
    "טרול",
    "זומבי",
    "פרשן",
    "נביא",
    "פנינה",
    "טבע",
    "אבק",
    "מיקרופון",
    "שידור",
    "גביע",
    "פרס",
]

BOARD_2_WORDS = [
    "סתיו",
    "קרב",
    "אביב",
    "עגיל",
    "קבוצה",
    "מקרה",
    "ביקור",
    "צעצוע",
    "שיתוף",
    "יין",
    "שקר",
    "במה",
    "אהדה",
    "חברה",
    "נצח",
    "צריח",
    "טיול",
    "יציאה",
    "פנינה",
    "טבע",
    "סוחר",
    "זאב",
    "צידה",
    "גביע",
    "כותונת",
]

BOARD_3_WORDS = [
    "ציבור",
    "אוטובוס",
    "ישראל",
    "מתח",
    "גס",
    "ברית",
    "גוש",
    "איום",
    "מורח",
    "קנה",
    "לידה",
    "מבחן",
    "אודם",
    "שוקו",
    "חטיף",
    "חוק",
    "רץ",
    "חצות",
    "רדיו",
    "כתם",
    "גרם",
    "כהן",
    "רושם",
    "אלמוג",
    "אופק",
]

BOARD1_CASE = ParseBoardTestCase(
    fixture_file="board1.jpg",
    language="heb",
    expected_words=BOARD_1_WORDS,
)

BOARD1_TOP_CASE = ParseBoardTestCase(
    fixture_file="board1_top.jpg",
    language="heb",
    expected_words=BOARD_1_WORDS,
    allowed_errors=0,
)

BOARD2_CASE = ParseBoardTestCase(
    fixture_file="board2.jpg",
    language="heb",
    expected_words=BOARD_2_WORDS,
    allowed_errors=4,
)

BOARD2_TOP_CASE = ParseBoardTestCase(
    fixture_file="board2_top.jpg",
    language="heb",
    expected_words=BOARD_2_WORDS,
    allowed_errors=3,
)

BOARD3_TILT_CASE = ParseBoardTestCase(
    fixture_file="board3_tilt.jpg",
    language="heb",
    expected_words=BOARD_3_WORDS,
)

BOARD3_TOP_CASE = ParseBoardTestCase(
    fixture_file="board3_top.jpg",
    language="heb",
    expected_words=BOARD_3_WORDS,
)

BOARD3_TOP2_CASE = ParseBoardTestCase(
    fixture_file="board3_top2.jpg",
    language="heb",
    expected_words=BOARD_3_WORDS,
    allowed_errors=2,
)
