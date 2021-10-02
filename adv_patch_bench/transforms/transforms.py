import math

import cv2 as cv
import numpy as np


def rotate(origin, point, angle):
    ox, oy = origin
    px, py = point
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def find_first_vertex(vertices):
    # Find the two left most vertices and select the top one
    left_two = np.argsort(vertices[:, 0])[:2]
    first_vertex = left_two[np.argsort(vertices[left_two, 1])[0]]
    return first_vertex


def sort_polygon_vertices(vertices):
    """
    Sort vertices such that the first one is the top left corner, and the rest
    follows in clockwise order. First, find a point inside the polygon (e.g., 
    mean of all vertices) and sort vertices by the angles.
    """
    # Compute normalized vectors from mean to vertices
    mean = vertices.mean(0)
    vec = vertices - mean
    # NOTE: y-coordinate has to be inverted because zero starts at the top
    vec[:, 1] *= -1
    vec_len = np.sqrt(vec[:, 0] ** 2 + vec[:, 1] ** 2)[:, None]
    vec /= vec_len
    # Compute angle from positive x-axis (negative sign is for clockwise)
    # NOTE: numpy.arctan2 takes y and x (not x and y)
    angles = - np.arctan2(vec[:, 1], vec[:, 0])
    sorted_idx = np.argsort(angles)
    vertices = vertices[sorted_idx]
    angles = angles[sorted_idx]

    first_idx = find_first_vertex(vertices)
    # If shape is diamond, find_first_vertex can be ambiguous
    if -np.pi * 5 / 8 < angles[first_idx] < -np.pi * 3 / 8 and len(vertices) == 4:
        first_idx = (first_idx - 1) % 4

    first = np.where(sorted_idx == first_idx)[0][0]
    sorted_idx = np.concatenate([sorted_idx[first:], sorted_idx[:first]])
    return vertices[sorted_idx]


def get_box_vertices(vertices, predicted_shape):
    """To apply perspective transform, we need to extract a set of four points
    from `vertices` of the polygon or the circle we identify. There is a 
    separate function for each `predicted_shape`.

    Args:
        vertices (np.ndarray): Array of vertices, shape: (num_vertices, 2)
        predicted_shape (str): Shape of the object

    Returns:
        np.ndarray: Array of vertices of a convex quadrilateral used for
            perspective transform
    """
    if predicted_shape == 'circle':
        vertices = get_box_from_ellipse(vertices)
    vertices = sort_polygon_vertices(vertices)
    box = vertices[BOX_VERTICES_IDX[predicted_shape]]
    assert box.shape == (4, 2)
    return box


def get_box_from_ellipse(rect):
    DEV_RATIO_THRES = 0.1
    assert len(rect) == 3
    # angle = rect[2] / 180 * np.pi
    # If width and height are close or angle is very large, the rotation may be
    # incorrectly estimated
    mean_size = (rect[1][0] + rect[1][1]) / 2
    dev_ratio = abs(rect[1][0] - mean_size) / mean_size
    if dev_ratio < DEV_RATIO_THRES:
        # angle = 0
        box = cv.boxPoints((rect[0], rect[1], 0.))
    else:
        box = cv.boxPoints(rect)
    # rect = np.array(rect[:2])
    # xmax, ymax = rect[0] + rect[1] / 2
    # xmin, ymin = rect[0] - rect[1] / 2
    # box = [[xmin, rect[0][1]], [rect[0][0], ymin], [xmax, rect[0][1]], [rect[0][0], ymax]]
    # # TODO: check rotation direction
    # # box = [rotate(rect[0], v, angle) for v in box]
    # # box = np.array([[int(v[0]), int(v[1])] for v in box], dtype=np.int64)
    # # box = np.array(box)
    # box = [rotate((rect[0][0], -rect[0][1]), (v[0], -v[1]), -angle) for v in box]
    # box = np.array([[int(v[0]), int(-v[1])] for v in box], dtype=np.int64)
    return box


BOX_VERTICES_IDX = {
    'circle': ((0, 1, 2, 3), ),
    'triangle_inverted': ((0, 1, 2, 3), ),
    'triangle': ((0, 1, 2, 3), ),
    'rect': ((0, 1, 2, 3), ),
    'diamond': ((0, 1, 2, 3), ),
    'pentagon': ((0, 2, 3, 4), ),
    'octagon': ((0, 2, 4, 6), ),
}
