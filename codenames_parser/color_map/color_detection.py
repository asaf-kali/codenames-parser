import logging
from typing import Type

import numpy as np
from codenames.generic.card import CardColor
from sklearn.cluster import KMeans

from codenames_parser.color_map.color_translator import (
    ColorTranslator,
    get_color_translator,
)
from codenames_parser.common.debug_util import SEPARATOR

log = logging.getLogger(__name__)


def classify_cell_colors[C: CardColor](cells: list[np.ndarray], color_type: Type[C]) -> list[C]:
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
    optimal_k = 4

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
        assigned_color = min(distances, key=distances.get)  # type: ignore
        cluster_to_color[i] = assigned_color
    log.info(f"Cluster to color mapping: {cluster_to_color}")
    return cluster_to_color
