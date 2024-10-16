import logging

import numpy as np
from codenames.game.color import CardColor
from sklearn.cluster import KMeans

from codenames_parser.color_map.consts import CARD_COLOR_TO_COLOR
from codenames_parser.color_map.models import Grid
from codenames_parser.common.debug_util import SEPARATOR

log = logging.getLogger(__name__)


def classify_cell_colors(cells: Grid[np.ndarray]) -> Grid[CardColor]:
    """
    Classifies the color of each cell by clustering their average colors.
    """
    log.info(SEPARATOR)
    log.info("Classifying cell colors using clustering...")

    # Flatten the grid and compute average colors
    avg_colors = np.empty((0, 3), dtype=np.float64)
    for cell_row in cells:
        for cell in cell_row:
            avg_color = cell.mean(axis=(0, 1))
            avg_colors = np.vstack([avg_colors, avg_color])

    # Determine the optimal number of clusters
    optimal_k = 4

    # Perform clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    labels = kmeans.fit_predict(avg_colors)

    # Map cluster labels to CardColor using predefined CODENAMES colors
    cluster_to_color = assign_colors_to_clusters(kmeans.cluster_centers_)

    # Reshape labels back to grid format
    card_colors: Grid[CardColor] = Grid(row_size=cells.row_size)
    idx = 0
    for cell_row in cells:
        row_labels = []
        for _ in cell_row:
            cluster_label = labels[idx]
            card_color = cluster_to_color[cluster_label]
            row_labels.append(card_color)
            idx += 1
        card_colors.append(row_labels)

    return card_colors


def assign_colors_to_clusters(cluster_centers: np.ndarray) -> dict:
    """
    Assigns CardColor to each cluster based on the closest CODENAMES color.
    """
    cluster_to_color = {}
    for i, center in enumerate(cluster_centers):
        distances = {}
        for card_color, codename_color in CARD_COLOR_TO_COLOR.items():
            # Compute the distance between the cluster center and the CODENAMES color
            distance = np.linalg.norm(center - codename_color.vector)
            distances[card_color] = distance
        # Find the CardColor with the minimum distance
        assigned_color = min(distances, key=distances.get)  # type: ignore
        cluster_to_color[i] = assigned_color
    log.info(f"Cluster to color mapping: {cluster_to_color}")
    return cluster_to_color