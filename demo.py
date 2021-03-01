import visualiser as vis
import waymo_utils as utils

import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--data_path', type=str, default='/media/harry/hdd/datasets/waymo', help='specify the path  of dataset')
    args = parser.parse_args()

    Waymo = utils.Waymo_data(args.data_path)

    assert len(Waymo) != 0, ("No data")


    sequence_idx = 200
    frame_idx = 100 #Each sequence has about 200 frames
    point_cloud = Waymo.get_lidar(sequence_idx,frame_idx)
    frame = Waymo.get_frame(sequence_idx,frame_idx)
    labels = Waymo.get_lidar_labels(frame)

    vis.vis_points_with_multi_boxes(labels['gt_boxes_lidar'],point_cloud)
