from typing import List

from codenames_parser.color_detection import classify_cell_colors
from codenames_parser.grid_detection import extract_cells
from codenames_parser.image_reader import read_image
from codenames_parser.perspective_correction import correct_perspective


def main(image_path: str) -> List[List[str]]:
    """
    Main function to process the Codenames board image.

    Args:
        image_path (str): Path to the input image.

    Returns:
        List[List[str]]: A 5x5 grid representing the colors of the cells.
    """
    image = read_image(image_path)
    corrected_image = correct_perspective(image)
    cells = extract_cells(corrected_image)
    grid_colors = classify_cell_colors(cells)

    return grid_colors


def entrypoint():
    import sys

    if len(sys.argv) != 2:
        print("Usage: python main.py <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]
    grid = main(image_path)
    for row in grid:
        print(row)


if __name__ == "__main__":
    entrypoint()
