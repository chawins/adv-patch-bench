from sklearn.cluster import KMeans
import numpy as np

# NOTE: Should we use L1 instead of Euclidean distance?
# NOTE: Should we cluster by 1D data instead?

COLORS = {
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'red': (255, 0, 0),
    'blue': (0, 0, 255),
    'yellow': (255, 204, 0),
    'orange': (255, 102, 0),
}


def run_kmean_single(img: np.ndarray, k: int, keep_channel: bool = True,
                     n_init: int = 10, max_iter: int = 300):
    kmean = KMeans(n_clusters=k, init='k-means++', n_init=n_init, max_iter=max_iter)
    img = img.reshape(-1, 1) if not keep_channel else img
    labels = kmean.fit_predict(img)
    return kmean.inertia_, kmean.cluster_centers_, labels


def run_kmean(img: np.ndarray, keep_channel: bool = True):
    loss_2, centers_2, labels_2 = run_kmean_single(img, 2, keep_channel=keep_channel)
    loss_3, centers_3, labels_3 = run_kmean_single(img, 3, keep_channel=keep_channel)
    n = img.shape[-1]
    is_2 = best_k(loss_2, loss_3, n)
    print('2' if is_2 else '3')
    centers = centers_2 if is_2 else centers_3
    labels = labels_2 if is_2 else labels_3
    return centers, labels


def best_k(loss_2: float, loss_3: float, n: int):
    # k selection is based on the score proposed by
    # https://www.ee.columbia.edu/~dpwe/papers/PhamDN05-kmeans.pdf
    alpha_2 = 1 - 3 / (4 * n)
    alpha_3 = alpha_2 + (1 - alpha_2) / 6
    score_2 = loss_2 / alpha_2
    score_3 = loss_3 / (alpha_3 * loss_2)
    return score_2 < score_3


def find_canonical_kmean(img: np.ndarray):
    centers, labels = run_kmean(img, keep_channel=True)

    white_idx = centers.sum(-1).argmin()
    black_idx = centers.sum(-1).argmax()
    white = centers[white_idx]
    black = centers[black_idx]
    beta = black
    alpha = white - beta

    canonical = centers[labels]
    return canonical, alpha, beta


def find_min_max(img: np.ndarray, method: str = 'percentile', q: float = 5.):
    if img.ndim == 1:
        img = img.reshape(-1, 1)
    assert method in ('percentile', 'kmean')
    if method == 'percentile':
        assert 0 <= q <= 100
        q = q if q < 50 else 100 - q
        min_ = np.percentile(img, q)
        max_ = np.percentile(img, 100 - q)

    if method == 'kmean':
        # Take top and bottom centers as max and min
        centers, _ = run_kmean(img, keep_channel=False)
        max_ = centers.max()
        min_ = centers.min()

    print(min_, max_)
    print(np.histogram(img))

    return min_, max_


def relight_range(img: np.ndarray):
    min_, max_ = find_min_max(img, method='percentile', q=10.)
    beta = min_
    alpha = max_ - beta
    return alpha, beta
