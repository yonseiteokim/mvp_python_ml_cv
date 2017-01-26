from PIL import Image
from numpy import *
from pylab import *


def normalize(points):
    """ Normalize a collection of points in Homogeneous coordinates so that last row = 1 """
    for row in points:
        row /= points[-1]

    return points


def make_homog(points):
    """ Convert a set of points to homogeneous coordinates """

    return vstack((points, ones((1, points.shape[1]))))


def H_from_points(fp, tp):
    if fp.shape != tp.shape:
        raise RuntimeError('number of points do not match')

    m = mean(fp[:2], axis=1)
    maxstd = max(std(fp[:2], axis=1)) + 1e-9
    C1 = diag([1 / maxstd, 1 / maxstd, 1])
    C1[0][2] = -m[0] / maxstd
    C1[1][2] = -m[1] / maxstd
    fp = dot(C1, fp)

    m = mean(tp[:2], axis=1)
    maxstd = max(std(tp[:2], axis=1)) + 1e-9
    C2 = diag([1 / maxstd, 1 / maxstd, 1])
    C2[0][2] = -m[0] / maxstd
    C2[1][2] = -m[1] / maxstd
    tp = dot(C2, tp)

    nbr_correspondences = fp.shape[1]
    A = zeros((2 * nbr_correspondences, 9))
    for i in range(nbr_correspondences):
        A[2 * i] = [-fp[0][i], -fp[1][i], -1, 0, 0, 0, tp[0][i] * fp[0][i], tp[0][i] * fp[1][i], tp[0][i]]
        A[2 * i + 1] = [0, 0, 0, -fp[0][i], -fp[1][i], -1, tp[1][i] * fp[0][i], tp[1][i] * fp[1][i], tp[1][i]]

    U, S, V = linalg.svd(A)
    H = V[8].reshape((3, 3))

    H = dot(linalg.inv(C2), dot(H, C1))

    return H / H[2, 2]


def Haffine_from_points(fp, tp):
    if fp.shape != tp.shape:
        raise RuntimeError('number of points do not match')

    m = mean(fp[:2], axis=1)
    maxstd = max(std(fp[:2], axis=1)) + 1e-9
    C1 = diag([1 / maxstd, 1 / maxstd, 1])
    C1[0][2] = -m[0] / maxstd
    C1[1][2] = -m[1] / maxstd
    fp_cond = dot(C1, fp)

    m = mean(tp[:2], axis=1)
    C2 = C1.copy()
    C2[0][2] = -m[0] / maxstd
    C2[1][2] = -m[1] / maxstd
    tp_cond = dot(C2, tp)

    A = concatenate((fp_cond[:2], tp_cond[:2]), axis=0)
    U, S, V = linalg.svd(A.T)

    tmp = V[:2].T
    B = tmp[:2]
    C = tmp[2:4]

    tmp2 = concatenate((dot(C, linalg.pinv(B)), zeros((2, 1))), axis=1)
    H = vstack((tmp2, [0, 0, 1]))
    H = dot(linalg.inv(C2), dot(H, C1))

    return H / H[2, 2]


class RansacModel(object):
    def __init__(self, debug=False):
        self.debug = debug

    def fit(self, data):
        data = data.T

        fp = data[:3, :4]
        tp = data[3:, :4]

        return H_from_points(fp, tp)

    def get_error(self, data, H):
        data = data.T
        fp = data[:3]
        tp = data[3:]

        fp_transformed = dot(H, fp)

        for i in range(3):
            fp_transformed[i] /= fp_transformed[2]

        return sqrt(sum((tp - fp_transformed) ** 2, axis=0))


def H_from_ransac(fp, tp, model, maxiter=1000, match_threshold=10):
    import ransac
    data = vstack((fp, tp))

    H, ransac_data = ransac.ransac(data.T, model, 4, maxiter, match_threshold, 10, return_all=True)
    return H, ransac_data['inliers']


































