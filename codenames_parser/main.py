from typing import List

from modules.color_detection import classify_cell_colors
from modules.grid_detection import extract_cells
from modules.image_reader import read_image
from modules.perspective_correction import correct_perspective


def main(image_path: str) -> List[List[str]]:
    """
    Main function to process the Codenames board image.

    Args:
        image_path (str): Path to the input image.

    Returns:
        List[List[str]]: A 5x5 grid representing the colors of the cells.
    """
    # Step 1: Read the image
    image = read_image(image_path)

    # Step 2: Correct perspective
    corrected_image = correct_perspective(image)

    # Step 3: Extract cells from the grid
    cells = extract_cells(corrected_image)

    # Step 4: Classify cell colors
    grid_colors = classify_cell_colors(cells)

    return grid_colors


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python main.py <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]
    grid = main(image_path)
    for row in grid:
        print(row)
