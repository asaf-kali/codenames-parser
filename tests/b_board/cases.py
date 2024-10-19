from typing import NamedTuple


class ParseBoardTestCase(NamedTuple):
    fixture_file: str
    language: str
    expected_words: list[str]


BOARD1_TOP_CASE = ParseBoardTestCase(
    fixture_file="board1_top.jpg",
    language="heb",
    expected_words=[
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
    ],
)

BOARD3_TOP_CASE = ParseBoardTestCase(
    fixture_file="board3_top.jpg",
    language="heb",
    expected_words=[
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
    ],
)
