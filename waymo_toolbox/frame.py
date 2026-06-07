import numpy as np
from typing import Dict, Optional

from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

from .utils import boxes_to_corners_3d


WAYMO_CLASSES = ["unknown", "Vehicle", "Pedestrian", "Sign", "Cyclist"]


class Frame:
    """Represents a single frame with point cloud, labels, and images."""

    def __init__(self, raw_frame):
        self._raw = raw_frame
        self._point_cloud = None
        self._labels = None

    @property
    def point_cloud(self) -> np.ndarray:
        """LiDAR point cloud as (N, 3) numpy array. Computed lazily."""
        if self._point_cloud is None:
            self._point_cloud = self._extract_point_cloud()
        return self._point_cloud

    @property
    def labels(self) -> Dict:
        """Object annotations as dict with arrays. Computed lazily."""
        if self._labels is None:
            self._labels = self._extract_labels()
        return self._labels

    @property
    def bounding_boxes(self) -> np.ndarray:
        """3D bounding boxes as (N, 7) array [x, y, z, l, w, h, heading]."""
        return self.labels["gt_boxes_lidar"]

    @property
    def box_corners(self) -> np.ndarray:
        """3D bounding box corners as (N, 8, 3) array."""
        return boxes_to_corners_3d(self.bounding_boxes)

    @property
    def location(self) -> str:
        return self._raw.context.stats.location

    @property
    def time_of_day(self) -> str:
        return self._raw.context.stats.time_of_day

    @property
    def weather(self) -> str:
        return self._raw.context.stats.weather

    @property
    def num_objects(self) -> int:
        return len(self.labels["name"])

    def __repr__(self):
        return (
            f"Frame(points={len(self.point_cloud)}, "
            f"objects={self.num_objects}, "
            f"location={self.location})"
        )

    def _extract_point_cloud(self) -> np.ndarray:
        (range_images, camera_projections, range_image_top_pose) = (
            frame_utils.parse_range_image_and_camera_projection(self._raw)
        )
        points, _ = frame_utils.convert_range_image_to_point_cloud(
            self._raw, range_images, camera_projections, range_image_top_pose
        )
        return np.concatenate(points, axis=0)

    def _extract_labels(self) -> Dict:
        obj_name, dimensions, locations, heading_angles = [], [], [], []
        difficulty, obj_ids, num_points = [], [], []

        for label in self._raw.laser_labels:
            box = label.box
            obj_name.append(WAYMO_CLASSES[label.type])
            dimensions.append([box.length, box.width, box.height])
            locations.append([box.center_x, box.center_y, box.center_z])
            heading_angles.append(box.heading)
            difficulty.append(label.detection_difficulty_level)
            obj_ids.append(label.id)
            num_points.append(label.num_lidar_points_in_box)

        annotations = {
            "name": np.array(obj_name),
            "dimensions": np.array(dimensions),
            "location": np.array(locations),
            "heading_angles": np.array(heading_angles),
            "difficulty": np.array(difficulty),
            "obj_ids": np.array(obj_ids),
            "num_points_in_gt": np.array(num_points),
        }

        # Filter out unknown class
        keep = annotations["name"] != "unknown"
        annotations = {k: v[keep] for k, v in annotations.items()}

        # Build GT boxes [x, y, z, l, w, h, heading]
        if len(annotations["name"]) > 0:
            annotations["gt_boxes_lidar"] = np.concatenate(
                [annotations["location"], annotations["dimensions"],
                 annotations["heading_angles"][..., np.newaxis]],
                axis=1,
            )
        else:
            annotations["gt_boxes_lidar"] = np.zeros((0, 7))

        return annotations
