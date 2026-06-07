import argparse
import logging

from .dataset import WaymoDataset
from .visualizer import create_visualizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Waymo Open Dataset Visualization Tool")
    parser.add_argument("--data-path", required=True, help="Path to Waymo dataset root")
    parser.add_argument("--segment", type=int, default=0, help="Segment index")
    parser.add_argument("--frame", type=int, default=0, help="Frame index within segment")
    parser.add_argument(
        "--backend", choices=["open3d", "mayavi"], default="open3d",
        help="Visualization backend"
    )
    args = parser.parse_args()

    dataset = WaymoDataset(args.data_path)
    logger.info(f"Dataset: {dataset}")

    frame = dataset.get_frame(args.segment, args.frame)
    logger.info(f"Frame: {frame}")

    vis = create_visualizer(args.backend)
    vis.show(frame.point_cloud, frame.bounding_boxes)


if __name__ == "__main__":
    main()
