import numpy as np


# Takes a point relative to the ChArUco board/world and transforms it
# into camera coordinates.
def transform_points(R, t, X_world):
    return X_world @ R.T + t


# Projects a 3D point into normalized 2D image coordinates.
def project_pinhole(X_cam):
    return X_cam[..., :2] / X_cam[..., 2:3]


# Distorts points to simulate where they would appear on an actual camera.
def distort(xy, dist):
    k1, k2, p1, p2, k3 = dist
    x = xy[..., 0]
    y = xy[..., 1]

    r2 = x**2 + y**2
    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3

    x_d = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x**2)
    y_d = y * radial + p1 * (r2 + 2 * y**2) + 2 * p2 * x * y

    return np.stack([x_d, y_d], axis=-1)
