import logging

import numpy as np
from codenames.classic.color import ClassicColor
from codenames.duet.card import DuetColor
from codenames.generic.card import CardColor
from sklearn.cluster import KMeans

from codenames_parser.color_map.color_translator import (
    ColorTranslator,
    get_color_translator,
)
from codenames_parser.common.utils.debug_util import SEPARATOR

log = logging.getLogger(__name__)


def classify_cell_colors[C: CardColor](cells: list[np.ndarray], color_type: type[C]) -> list[C]:
    """
    Classifies the color of each cell by clustering their average colors.
    """
    log.info(SEPARATOR)
    log.info("Classifying cell colors using clustering...")

    # Flatten the grid and compute average colors
    avg_colors = np.empty((0, 3), dtype=np.float64)
    for cell in cells:
        avg_color = cell.mean(axis=(0, 1))
        avg_colors = np.vstack([avg_colors, avg_color])

    # Determine the optimal number of clusters
    optimal_k = _get_optimal_k(color_type)

    # Perform clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    labels = kmeans.fit_predict(avg_colors)

    # Map cluster labels to CardColor using predefined CODENAMES colors
    color_translator = get_color_translator(color_type=color_type)
    cluster_to_color = assign_colors_to_clusters(kmeans.cluster_centers_, color_translator=color_translator)

    # Reshape labels back to grid format
    card_colors: list[C] = []
    for i in range(len(cells)):
        cluster_label = labels[i]
        card_color = cluster_to_color[cluster_label]
        card_colors.append(card_color)
    return card_colors


def _get_optimal_k(color_type: type[CardColor]) -> int:
    """
    Returns the optimal number of clusters for the given color type.
    """
    if color_type == ClassicColor:
        return 4
    if color_type == DuetColor:
        return 3
    raise NotImplementedError(f"Unsupported color type: {color_type}")


def assign_colors_to_clusters(cluster_centers: np.ndarray, color_translator: ColorTranslator) -> dict:
    """
    Assigns CardColor to each cluster based on the closest CODENAMES color.
    """
    cluster_to_color = {}
    for i, center in enumerate(cluster_centers):
        distances: dict[str, float] = {}
        for card_color, codename_color in color_translator.items():
            # Compute the distance between the cluster center and the CODENAMES color
            distance = np.linalg.norm(center - codename_color.vector)
            distances[card_color] = float(distance)
        # Find the CardColor with the minimum distance
        center_rgb = int(center[2]), int(center[1]), int(center[0])
        log.debug(f"Cluster {i} center color: {center_rgb}, distances: {distances}")
        assigned_color = min(distances, key=distances.get)  # type: ignore
        cluster_to_color[i] = assigned_color
    log.info(f"Cluster to color mapping: {cluster_to_color}")
    return cluster_to_color
