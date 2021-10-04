def draw_from_contours(img, contours, color=[0, 0, 255, 255]):
    if not isinstance(contours, list):
        contours = [contours]
    for contour in contours:
        if contour.ndim == 3:
            contour_coord = (contour[:, 0, 1], contour[:, 0, 0])
        else:
            contour_coord = (contour[:, 1], contour[:, 0])
        img[contour_coord] = color
    return img
