import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tensorflow.compat.v1 as tf

tf.enable_eager_execution()
from waymo_open_dataset import dataset_pb2 as open_dataset

from .frame import Frame

logger = logging.getLogger(__name__)


class Segment:
    """Represents a single driving segment (one TFRecord file)."""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.name = file_path.name
        self._dataset = None

    def __getitem__(self, frame_idx: int) -> Frame:
        dataset = tf.data.TFRecordDataset(str(self.file_path), compression_type="")
        for index, data in enumerate(dataset):
            if index == frame_idx:
                raw_frame = open_dataset.Frame()
                raw_frame.ParseFromString(bytearray(data.numpy()))
                return Frame(raw_frame)
        raise IndexError(f"Frame index {frame_idx} out of range for segment {self.name}")

    def __repr__(self):
        return f"Segment({self.name})"


class WaymoDataset:
    """Main entry point for loading and accessing Waymo Open Dataset."""

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self._segments = self._discover_segments()
        logger.info(f"Loaded dataset with {self.num_segments} segments from {self.data_path}")

    @classmethod
    def from_config(cls, config_path: str):
        """Create dataset from a JSON config file."""
        with open(config_path) as f:
            config = json.load(f)
        return cls(config["data_path"])

    @property
    def num_segments(self) -> int:
        return len(self._segments)

    def __len__(self) -> int:
        return self.num_segments

    def __getitem__(self, segment_idx: int) -> Segment:
        if segment_idx >= len(self._segments):
            raise IndexError(f"Segment index {segment_idx} out of range (total: {self.num_segments})")
        return Segment(self._segments[segment_idx])

    def __iter__(self):
        for seg_path in self._segments:
            yield Segment(seg_path)

    def __repr__(self):
        return f"WaymoDataset({self.data_path}, segments={self.num_segments})"

    def get_frame(self, segment_idx: int, frame_idx: int) -> Frame:
        """Get a single frame from a specific segment."""
        return self[segment_idx][frame_idx]

    def load_frames_parallel(
        self, requests: List[Tuple[int, int]], max_workers: int = 4
    ) -> List[Frame]:
        """Load multiple frames in parallel.

        Args:
            requests: List of (segment_idx, frame_idx) tuples
            max_workers: Number of parallel threads for I/O
        """
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(self.get_frame, seg, frame) for seg, frame in requests]
            return [f.result() for f in futures]

    def _discover_segments(self) -> List[Path]:
        raw_data_path = self.data_path / "raw_data"
        if not raw_data_path.exists():
            raise FileNotFoundError(f"raw_data directory not found at {raw_data_path}")
        segments = sorted(raw_data_path.iterdir())
        return [s for s in segments if s.is_file()]
