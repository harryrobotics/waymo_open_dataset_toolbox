import numpy as np


def rotate_points_along_z(points: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Rotate points along the z-axis.

    Args:
        points: (B, N, 3+C) array of points
        angle: (B,) array of angles in radians
    """
    cosa = np.cos(angle)
    sina = np.sin(angle)
    zeros = np.zeros(points.shape[0])
    ones = np.ones(points.shape[0])

    rot_matrix = np.stack(
        (cosa, sina, zeros, -sina, cosa, zeros, zeros, zeros, ones), axis=1
    ).reshape(-1, 3, 3)

    points_rot = np.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = np.concatenate((points_rot, points[:, :, 3:]), axis=-1)
    return points_rot


def boxes_to_corners_3d(boxes3d: np.ndarray) -> np.ndarray:
    """Convert 3D bounding boxes to 8 corner points.

    Args:
        boxes3d: (N, 7) array [x, y, z, dx, dy, dz, heading]

    Returns:
        corners: (N, 8, 3) array of corner coordinates

    Corner ordering:
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    """
    template = (
        np.array(
            [
                [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
                [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
            ]
        )
        / 2
    )

    corners3d = np.repeat(boxes3d[:, None, 3:6], 8, axis=1) * template[None, :, :]
    corners3d = rotate_points_along_z(
        corners3d.reshape((-1, 8, 3)), boxes3d[:, 6]
    ).reshape((-1, 8, 3))
    corners3d += boxes3d[:, None, 0:3]

    return corners3d
