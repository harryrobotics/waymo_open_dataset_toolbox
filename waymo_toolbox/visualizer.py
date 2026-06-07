from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from .utils import boxes_to_corners_3d


class BaseVisualizer(ABC):
    """Abstract base class for point cloud visualization backends."""

    @abstractmethod
    def show(self, points: np.ndarray, boxes: Optional[np.ndarray] = None):
        """Display point cloud with optional bounding boxes.

        Args:
            points: (N, 3) point cloud array
            boxes: (M, 7) bounding boxes [x, y, z, l, w, h, heading]
        """
        pass


class Open3DVisualizer(BaseVisualizer):
    """3D visualization using Open3D."""

    def __init__(self, box_color=(0, 0, 1)):
        self.box_color = list(box_color)

    def show(self, points: np.ndarray, boxes: Optional[np.ndarray] = None):
        import open3d as o3d

        geometries = []

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        geometries.append(pcd)

        if boxes is not None and len(boxes) > 0:
            corners = boxes_to_corners_3d(boxes)
            for box_corners in corners:
                line_set = self._create_box_lines(box_corners)
                geometries.append(line_set)

        o3d.visualization.draw_geometries(geometries)

    def _create_box_lines(self, corners: np.ndarray):
        import open3d as o3d

        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [0, 4], [1, 5], [2, 6], [3, 7],
            [4, 5], [5, 6], [6, 7], [7, 4],
        ]
        colors = [self.box_color for _ in range(len(lines))]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return line_set


class MayaviVisualizer(BaseVisualizer):
    """3D visualization using Mayavi."""

    def show(self, points: np.ndarray, boxes: Optional[np.ndarray] = None):
        from mayavi import mlab

        mlab.figure(bgcolor=(0, 0, 0))
        mlab.points3d(
            points[:, 0], points[:, 1], points[:, 2],
            mode="point", colormap="gnuplot",
        )

        if boxes is not None and len(boxes) > 0:
            corners = boxes_to_corners_3d(boxes)
            for box_corners in corners:
                self._draw_box(mlab, box_corners)

        mlab.show()

    def _draw_box(self, mlab, corners: np.ndarray):
        connections = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [0, 4], [1, 5], [2, 6], [3, 7],
            [4, 5], [5, 6], [6, 7], [7, 4],
        ]
        for start, end in connections:
            mlab.plot3d(
                [corners[start, 0], corners[end, 0]],
                [corners[start, 1], corners[end, 1]],
                [corners[start, 2], corners[end, 2]],
                color=(0, 1, 0), tube_radius=None,
            )


VISUALIZER_MAP = {
    "open3d": Open3DVisualizer,
    "mayavi": MayaviVisualizer,
}


def create_visualizer(backend: str = "open3d") -> BaseVisualizer:
    """Factory to create a visualizer by backend name.

    Args:
        backend: "open3d" or "mayavi"
    """
    if backend not in VISUALIZER_MAP:
        raise ValueError(f"Unknown backend '{backend}'. Choose from: {list(VISUALIZER_MAP.keys())}")
    return VISUALIZER_MAP[backend]()
