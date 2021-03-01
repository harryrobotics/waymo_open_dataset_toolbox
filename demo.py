import numpy as np
import pickle
import visualiser as vis
import waymo_utils as utils
from pathlib import Path
from os import walk

ROOT_DIR = Path('/media/harry/hdd/datasets/waymo')

list_file = []
for (dirpath, dirnames, filenames) in walk(ROOT_DIR / 'raw_data'):
    list_file.extend(filenames)
# print(val_sequences)
print("Number of sequences: ", len(list_file))

sequence_idx = 700
sample_idx = 20

print(sequence_idx)
print(sample_idx)


sequence_name = list_file[sequence_idx]

print(sequence_name)

FILENAME = str(ROOT_DIR / 'raw_data' / sequence_name)

segment = utils.get_data_segment(FILENAME)


frame = utils.get_frame_from_index(segment,sample_idx)

utils.print_frame_info(frame)

lidar_labels = utils.get_labels_from_frame(frame)

point_cloud,_ = utils.point_cloud_from_frame(frame)

#Test visualise with open3d
utils.render_camera(frame,cam_name='all')

utils.render_camera(frame,cam_name='FRONT')

vis.vis_points_with_multi_boxes(lidar_labels['gt_boxes_lidar'],point_cloud)

#Test visualise with mayavi
# detection_path = './waymo_pp_result.pkl'
#
# with open(detection_path, 'rb') as f:
#     dts = pickle.load(f)
#
# dts_sample = dts[sequence_idx]
# print(dts_sample)

# import mayavi_visualiser as V
# import mayavi.mlab as mlab
#
# V.draw_scenes(point_cloud,lidar_labels['gt_boxes_lidar'],dts_sample['boxes_lidar'],
#                 dts_sample['score']) #Can not draw 1 single box
# mlab.show(stop=True)
