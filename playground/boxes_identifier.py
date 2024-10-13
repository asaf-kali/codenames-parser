from typing import NamedTuple

import cv2
import numpy as np

# Based on ChatGPT translation of: https://stackoverflow.com/questions/57005217/detecting-boxes-via-hough-transform


class Box(NamedTuple):
    p1: np.ndarray
    p2: np.ndarray


def intersection(o1: np.ndarray, p1: np.ndarray, o2: np.ndarray, p2: np.ndarray) -> np.ndarray | None:
    """
    finds the intersection point of two lines defined by points (o1, p1) and (o2, p2).

    args:
        o1: starting point of the first line.
        p1: ending point of the first line.
        o2: starting point of the second line.
        p2: ending point of the second line.

    returns:
        the intersection point as a num_py array, or None if lines are parallel.
    """
    x = o2 - o1
    d1 = p1 - o1
    d2 = p2 - o2

    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-8:
        return None

    t1 = (x[0] * d2[1] - x[1] * d2[0]) / cross
    r = o1 + d1 * t1
    return r


def cluster_pts(input_pts: list[np.ndarray], cluster_radius_squared: float) -> list[np.ndarray]:
    """
    clusters a list of points based on a radius threshold.

    args:
        input_pts: list of points to cluster.
        cluster_radius_squared: square of the clustering radius.

    returns:
        a list of cluster centers.
    """
    output_pts = []
    input_pts = input_pts.copy()
    while len(input_pts) > 0:
        cluster_center = input_pts[0]
        while True:
            new_clust_center = np.zeros(2)
            averaging_count = 0
            cluster_indices = []
            for i in range(len(input_pts)):
                if (
                    cluster_radius_squared
                    >= (input_pts[i][0] - cluster_center[0]) ** 2 + (input_pts[i][1] - cluster_center[1]) ** 2
                ):
                    new_clust_center += input_pts[i]
                    averaging_count += 1
                    cluster_indices.append(i)
            new_clust_center = new_clust_center / averaging_count

            if np.allclose(new_clust_center, cluster_center, atol=1e-8):
                # remove clustered points and store cluster center
                input_pts = [input_pts[i] for i in range(len(input_pts)) if i not in cluster_indices]
                output_pts.append(cluster_center)
                break
            else:
                cluster_center = new_clust_center
    return output_pts


def find_second_point(corners: list[np.ndarray], first_pt: np.ndarray, box_side_length_guess: float) -> int:
    """Find a point vertically aligned with the first point."""
    for i in range(1, len(corners)):
        if abs(corners[i][0] - first_pt[0]) < box_side_length_guess / 2.0:
            return i
    return -1


def find_third_points(
    corners: list[np.ndarray], second_pt_index: int, box_side_length_guess: float, approx_box_size: float
) -> tuple[int, int, float, float]:
    """Find points to the left and right of the second point at the same vertical level."""
    third_index_left = -1
    third_index_right = -1
    min_dist_right = approx_box_size
    min_dist_left = -approx_box_size

    second_pt = corners[second_pt_index]

    for i in range(2, len(corners)):
        if abs(corners[i][1] - second_pt[1]) < box_side_length_guess / 2.0:
            dist = corners[i][0] - second_pt[0]
            if 0 > dist > min_dist_left:
                min_dist_left = dist
                third_index_left = i
            elif 0 < dist < min_dist_right:
                min_dist_right = dist
                third_index_right = i

    return third_index_left, third_index_right, min_dist_left, min_dist_right


def find_fourth_points(
    corners: list[np.ndarray], third_index_left: int, third_index_right: int, box_side_length_guess: float
) -> tuple[int, int]:
    """Find fourth points to complete the boxes."""
    fourth_index_left = -1
    fourth_index_right = -1

    for i in range(1, len(corners)):
        if i in (third_index_left, third_index_right):
            continue
        if third_index_left != -1 and abs(corners[i][0] - corners[third_index_left][0]) < box_side_length_guess / 2.0:
            fourth_index_left = i
        if third_index_right != -1 and abs(corners[i][0] - corners[third_index_right][0]) < box_side_length_guess / 2.0:
            fourth_index_right = i

    return fourth_index_left, fourth_index_right


def compute_boxes(
    corners: list[np.ndarray],
    first_pt: np.ndarray,
    third_index_left: int,
    third_index_right: int,
    fourth_index_left: int,
    fourth_index_right: int,
    shrink_boxes: bool,
) -> list[Box]:
    """Compute the boxes based on found indices."""
    boxes = []
    if not shrink_boxes:
        if fourth_index_right != -1:
            box = Box(first_pt, corners[third_index_right])
            boxes.append(box)
        if fourth_index_left != -1:
            box = Box(first_pt, corners[third_index_left])
            boxes.append(box)
    else:
        if fourth_index_right != -1:
            box_p1 = first_pt * 0.90 + corners[third_index_right] * 0.10
            box_p2 = first_pt * 0.10 + corners[third_index_right] * 0.90
            box = Box(box_p1, box_p2)
            boxes.append(box)
        if fourth_index_left != -1:
            box_p1 = first_pt * 0.90 + corners[third_index_left] * 0.10
            box_p2 = first_pt * 0.10 + corners[third_index_left] * 0.90
            box = Box(box_p1, box_p2)
            boxes.append(box)
    return boxes


def find_boxes(corners: list[np.ndarray], shrink_boxes: bool = False, box_side_length_guess: int = 50) -> list[Box]:
    """
    Finds rectangular boxes from corner points.

    Args:
        corners: list of corner points.
        shrink_boxes: Whether to shrink the boxes slightly.
        box_side_length_guess: Initial guess for the box side length.

    Returns:
        A list of Box representing boxes as pairs of points.
    """
    out_boxes: list[Box] = []
    approx_box_size = 1000 * box_side_length_guess
    corners = corners.copy()
    while len(corners) > 4:
        first_pt = corners[0]
        second_pt_index = find_second_point(corners, first_pt, box_side_length_guess)
        if second_pt_index == -1:
            print("bad box point tossed")
            corners.pop(0)
            continue

        (
            third_index_left,
            third_index_right,
            min_dist_left,
            min_dist_right,
        ) = find_third_points(corners, second_pt_index, box_side_length_guess, approx_box_size)

        if third_index_left != -1:
            approx_box_size = 1.5 * abs(min_dist_left)
        if third_index_right != -1:
            approx_box_size = 1.5 * min_dist_right

        fourth_index_left, fourth_index_right = find_fourth_points(
            corners, third_index_left, third_index_right, box_side_length_guess
        )

        boxes = compute_boxes(
            corners,
            first_pt,
            third_index_left,
            third_index_right,
            fourth_index_left,
            fourth_index_right,
            shrink_boxes,
        )
        out_boxes.extend(boxes)

        # Remove used points
        if second_pt_index > 0:
            corners.pop(second_pt_index)
        corners.pop(0)

    print(approx_box_size)
    return [Box(p1=box[0], p2=box[1]) for box in out_boxes]


def main():  # noqa: C901
    image = cv2.imread("write.png", cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("error loading image")
        return

    cv2.imshow("source", image)

    edges = cv2.Canny(image, 50, 200, apertureSize=3)
    line_overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    corner_overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    final_boxes = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    print(f"{image.shape[1]} , {image.shape[0]}")

    # parameters for hough_lines
    rho_res = 5
    theta_res = np.pi / 180
    threshold_h = int(2 * edges.shape[1] * 0.6)
    threshold_v = int(2 * edges.shape[0] * 0.6)

    # detect horizontal lines
    lines_horizontal = cv2.HoughLines(
        edges, rho_res, theta_res, threshold=threshold_h, min_theta=np.pi / 4, max_theta=3 * np.pi / 4
    )

    # detect vertical lines
    lines_vertical = cv2.HoughLines(
        edges, rho_res, theta_res, threshold=threshold_v, min_theta=-np.pi / 32, max_theta=np.pi / 32
    )

    pts_lh = []
    if lines_horizontal is not None:
        for line in lines_horizontal:
            pt1, pt2 = add_line_points(line, pts_lh)
            cv2.line(line_overlay, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)

    pts_lv = []
    if lines_vertical is not None:
        for line in lines_vertical:
            pt1, pt2 = add_line_points(line, pts_lv)
            cv2.line(line_overlay, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow("edged", edges)
    cv2.imshow("detected lines", line_overlay)

    # find intersections
    x_pts = []
    for i in range(len(pts_lh) // 2):
        for j in range(len(pts_lv) // 2):
            o1 = pts_lh[2 * i].astype(float)
            p1 = pts_lh[2 * i + 1].astype(float)
            o2 = pts_lv[2 * j].astype(float)
            p2 = pts_lv[2 * j + 1].astype(float)
            x_pt = intersection(o1, p1, o2, p2)
            if x_pt is not None:
                x_pts.append(x_pt)

    # cv2.waitKey(1000)

    # cluster intersection points
    box_corners = cluster_pts(x_pts, 25 * 25)
    for corner in box_corners:
        cv2.circle(corner_overlay, (int(corner[0]), int(corner[1])), 5, (0, 255, 0), 2)

    cv2.imshow("detected corners", corner_overlay)

    # find and draw boxes
    ocr_boxes = find_boxes(box_corners, shrink_boxes=True)
    for i, box in enumerate(ocr_boxes):
        p1 = (int(box[0][0]), int(box[0][1]))
        p2 = (int(box[1][0]), int(box[1][1]))
        color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)][i % 3]
        cv2.rectangle(final_boxes, p1, p2, color, 2)

    cv2.imshow("detected boxes", final_boxes)

    cv2.waitKey(0)


def add_line_points(line: np.ndarray, points_list: list[np.ndarray]) -> tuple[tuple[int, int], tuple[int, int]]:
    rho, theta = line[0]
    a, b = np.cos(theta), np.sin(theta)
    x0, y0 = a * rho, b * rho
    pt1 = (int(round(x0 + 1000 * (-b))), int(round(y0 + 1000 * a)))
    pt2 = (int(round(x0 - 1000 * (-b))), int(round(y0 - 1000 * a)))
    points_list.append(np.array(pt1))
    points_list.append(np.array(pt2))
    return pt1, pt2


if __name__ == "__main__":
    main()
