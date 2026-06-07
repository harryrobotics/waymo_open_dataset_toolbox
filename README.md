# Waymo Open Dataset Toolbox

A Python toolkit for loading, processing, and visualizing 3D LiDAR point clouds and annotations from the [Waymo Open Dataset](https://waymo.com/open/).

## Installation

```bash
pip install -e .

# With Open3D visualization:
pip install -e ".[open3d]"

# With Mayavi visualization:
pip install -e ".[mayavi]"
```

## Quick Start

### Python API

```python
from waymo_toolbox import WaymoDataset, create_visualizer

# Load dataset
dataset = WaymoDataset("/path/to/waymo")
print(dataset)  # WaymoDataset(/path/to/waymo, segments=798)

# Access a frame
frame = dataset[0][100]  # segment 0, frame 100
print(frame)  # Frame(points=177042, objects=34, location=location_sf)

# Visualize
vis = create_visualizer("open3d")
vis.show(frame.point_cloud, frame.bounding_boxes)

# Load multiple frames in parallel
frames = dataset.load_frames_parallel([
    (0, 0), (0, 50), (1, 0), (1, 50)
], max_workers=4)
```

### CLI

```bash
waymo-vis --data-path /path/to/waymo --segment 0 --frame 100 --backend open3d
```

## Dataset Structure

Download the Waymo Open Dataset (~1TB) and organize as:

```
waymo/
├── raw_data/
│   ├── segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord
│   ├── segment-...
├── LICENSE
```

## Architecture

```
waymo_toolbox/
├── __init__.py      # Public API: WaymoDataset, Frame, create_visualizer
├── dataset.py       # WaymoDataset, Segment classes (data loading + threading)
├── frame.py         # Frame class (point cloud, labels, bounding boxes)
├── visualizer.py    # BaseVisualizer ABC + Open3D/Mayavi implementations (Factory)
├── utils.py         # Geometry utilities (rotation, box corners)
└── cli.py           # Command-line interface
```

## Examples

LiDAR with ground truth bounding boxes:

![](https://github.com/harryrobotics/waymo_open_dataset_toolbox/blob/master/media/lidar.png)

![](https://github.com/harryrobotics/waymo_open_dataset_toolbox/blob/master/media/lidar1.png)

Front camera:

![](https://github.com/harryrobotics/waymo_open_dataset_toolbox/blob/master/media/front_camera.png)

5 cameras:

![](https://github.com/harryrobotics/waymo_open_dataset_toolbox/blob/master/media/all_cameras.png)
