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
    return vertices.reshape(-1, 2)


def get_shape_from_vertices(vertices):
    num_vertices = len(vertices)
    if num_vertices == 3:
        return 'triangle'
    if num_vertices == 4:
        return 'rect'
    if num_vertices == 5:
        return 'pentagon'
    if num_vertices == 8:
        return 'octagon'
    return 'other'
