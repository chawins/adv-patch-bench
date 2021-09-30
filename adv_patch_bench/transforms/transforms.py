from imutils import perspective
import numpy as np


def sort_box_vertices(box):
    """
    Arrange vertices such that the first one is the top left corner and go in
    the clockwise order.
    """
    # Find the two left most vertices and select the top one
    v = np.array(box)
    # left_two = np.argsort(v[:, 0])[:2]
    # first_vertex = left_two[np.argsort(v[left_two, 1])[0]]
    out = perspective.order_points(box)

    # print(first_vertex)
    # print(predicted_shape)
    print(v)

    import pdb
    pdb.set_trace()


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
    box = BOX_FUNC[predicted_shape](vertices)
    return sort_box_vertices(box)


def get_box_from_circle(rect):
    pass


def get_box_from_inv_triangle(vertices):
    pass


def get_box_from_triangle(vertices):
    pass


def get_box_from_rect(vertices):
    return vertices


def get_box_from_pentagon(vertices):
    pass


def get_box_from_octagon(vertices):
    pass


def get_transform_matrix(img, mask, predicted_shape):
    pass


BOX_FUNC = {
    'circle': get_box_from_circle,
    'inv_triangle': get_box_from_inv_triangle,
    'triangle': get_box_from_triangle,
    'rect': get_box_from_rect,
    'pentagon': get_box_from_pentagon,
    'octagon': get_box_from_octagon,
}
