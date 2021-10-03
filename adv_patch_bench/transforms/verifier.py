import cv2 as cv
import numpy as np

POLYGON_ERROR = 0.04


def detect_polygon(contour):
    eps = cv.arcLength(contour, True) * POLYGON_ERROR
    vertices = cv.approxPolyDP(contour, eps, True)
    return vertices


def get_corners(mask):
    # Check that we have a binary mask
    assert mask.ndim == 2
    assert (mask == 1).sum() + (mask == 0).sum() == mask.shape[0] * mask.shape[1]

    # Find contour of the object
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # Find convex hull to combine multiple contours and/or fix some occlusion
    cat_contours = np.concatenate(contours, axis=0)
    hull = cv.convexHull(cat_contours, returnPoints=True)

    # Fit polygon to remove some annotation errors and get vertices
    vertices = detect_polygon(hull)

    # vertices: (distance from left edge, distance from top edge)
    return vertices.reshape(-1, 2), hull


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
        # We want first_idx for diamond to be the left corner
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
    box = vertices[SHAPE_TO_VERTICES[predicted_shape]]
    assert box.shape == (4, 2)
    return box


def get_shape_from_vertices(vertices):
    num_vertices = len(vertices)
    vertices = sort_polygon_vertices(vertices)
    height = vertices[:, 1].max() - vertices[:, 1].min()
    if num_vertices == 3:
        if abs(vertices[0, 1] - vertices[1, 1]) / height < 0.5:
            shape = 'triangle_inverted'
        else:
            shape = 'triangle'
    elif num_vertices == 4:
        if abs(vertices[0, 1] - vertices[2, 1]) / height < 0.5:
            shape = 'diamond'
        else:
            shape = 'rect'
    elif num_vertices == 5:
        shape = 'pentagon'
    elif num_vertices == 8:
        shape = 'octagon'
    else:
        shape = 'other'
    return shape
