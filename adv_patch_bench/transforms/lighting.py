from sklearn.cluster import KMeans
import numpy as np

# NOTE: Should we use L1 instead of Euclidean distance?
# NOTE: Should we cluster by 1D data instead?

COLORS = {
    'white': (255, 255, 255),
    'black': (255, 255, 255),
    'red': (255, 0, 0),
    'blue': (0, 0, 255),
    'yellow': (255, 204, 0),
    'orange': (255, 102, 0),
}


def run_kmean(img, k, n_init=10, max_iter=300):
    kmean = KMeans(n_clusters=k, init='k-means++', n_init=n_init, max_iter=max_iter)
    labels = kmean.fit_predict(img.reshape(-1, 3))
    return kmean.inertia_, kmean.cluster_centers_, labels


def best_k(loss_2, loss_3, n):
    alpha_2 = 1 - 3 / (4 * n)
    alpha_3 = alpha_2 + (1 - alpha_2) / 6
    score_2 = loss_2 / alpha_2
    score_3 = loss_3 / (alpha_3 * loss_2)
    return score_2 < score_3


def find_canonical(img: np.ndarray):
    loss_2, centers_2, labels_2 = run_kmean(img, 2)
    loss_3, centers_3, labels_3 = run_kmean(img, 3)
    n = img.shape[-1]
    is_2 = best_k(loss_2, loss_3, n)
    centers = centers_2 if is_2 else centers_3
    labels = labels_2 if is_2 else labels_3

    white_idx = centers.sum(-1).argmin()
    black_idx = centers.sum(-1).argmax()
    white = centers[white_idx]
    black = centers[black_idx]
    beta = black
    alpha = white - beta

    canonical = centers[labels]
    return canonical, alpha, beta
