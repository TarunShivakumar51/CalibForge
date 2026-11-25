import numpy as np


# Takes a point relative to the ChArUco board/world and transforms it into camera points.
def transform_points(R, t, X_world):
    t = t.reshape(3)
    return X_world @ R.T + t


# Projects a 3D point into normalized 2D image points.
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


# Convert normalized points to pixels relative to image's top-left corner
def to_pixels(xy, K):
    fx = K[0, 0]
    cx = K[0, 2]

    fy = K[1, 1]
    cy = K[1, 2]

    u = fx * xy[..., 0] + cx
    v = fy * xy[..., 1] + cy

    return np.stack([u, v], axis=-1)


def project(R, t, X_world, dist, K):
    X_cam = transform_points(R, t, X_world)
    normalized_xy = project_pinhole(X_cam)
    distorted_xy = distort(normalized_xy, dist)
    pixel = to_pixels(distorted_xy, K)

    return pixel


def undistort(xy_d, dist, iters=20):
    k1, k2, p1, p2, k3 = dist
    x_d, y_d = xy_d[..., 0], xy_d[..., 1]
    x, y = x_d, y_d

    for _ in range(iters):
        r2 = x**2 + y**2
        radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
        dx = 2 * p1 * x * y + p2 * (r2 + 2 * x**2)
        dy = p1 * (r2 + 2 * y**2) + 2 * p2 * x * y
        x, y = (x_d - dx) / radial, (y_d - dy) / radial

    return np.stack([x, y], axis=-1)


def unproject(uv, K, dist):

    fx = K[0, 0]
    cx = K[0, 2]

    fy = K[1, 1]
    cy = K[1, 2]

    u = uv[..., 0]
    v = uv[..., 1]

    x_d = (u - cx) / fx
    y_d = (v - cy) / fy

    return undistort(np.stack([x_d, y_d], axis=-1), dist)
