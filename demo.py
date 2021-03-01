import numpy as np
import pickle
import visualiser as vis
import waymo_utils as utils
from pathlib import Path
from os import walk


Waymo = utils.Waymo_data('/media/harry/hdd/datasets/waymo')

sequence_idx = 200
frame_idx = 100 #Each sequence has about 200 frames
point_cloud = Waymo.get_lidar(sequence_idx,frame_idx)
frame = Waymo.get_frame(sequence_idx,frame_idx)
labels = Waymo.get_lidar_labels(frame)

vis.vis_points_with_multi_boxes(labels['gt_boxes_lidar'],point_cloud)
