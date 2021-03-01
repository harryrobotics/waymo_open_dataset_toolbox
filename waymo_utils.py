#Some code is written by myself
#Some code is written by Google Waymo Open dataset team
#https://github.com/waymo-research/waymo-open-dataset
#Some code is written by teams from MMLab (Github OpenPCDet)
# https://github.com/open-mmlab/OpenPCDet

import copy
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


import tensorflow.compat.v1 as tf
import itertools
tf.enable_eager_execution()
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

from pathlib import Path

class Waymo_data():

    def __init__(self,data_path):
        self.data_path = Path(data_path)
        self.segment_list = self.get_segment_list()

    def get_segment_list(self):
        from os import walk
        list_file = []
        for (dirpath, dirnames, filenames) in walk(self.data_path / 'raw_data'):
            list_file.extend(filenames)
            # print(val_sequences)
            print("Number of sequences: ", len(list_file))
        return list_file

    def get_frame(self,segment_idx,sample_idx):
        sequence_name = self.segment_list[segment_idx]
        file_name = str(self.data_path  / 'raw_data' / sequence_name)
        segment = get_data_segment(file_name)
        frame = get_frame_from_index(segment,sample_idx)

        return frame

    def get_lidar(self,segment_idx,sample_idx):
        frame = self.get_frame(segment_idx,sample_idx)
        point_cloud,_ = point_cloud_from_frame(frame)

        return point_cloud

    def get_lidar_labels(self,frame):
        labels = get_labels_from_frame(frame)

        return labels


def print_frame_info(frame):
    print("Location: ",frame.context.stats.location)
    print("Time: ",frame.context.stats.time_of_day)
    print("Weather: ",frame.context.stats.weather)
    print("Labels count: ", frame.context.stats.laser_object_counts)


def get_data_segment(raw_file_name):
    return tf.data.TFRecordDataset(raw_file_name, compression_type='')

def get_frame_from_index(segment,frame_index):
    for index,data in enumerate(segment):
        if index ==frame_index:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            return frame
    return None

def point_cloud_from_frame(frame):
    '''
    Args:
    Ins: data frame read tfrecord
    Outs: points aggerated, points top, front, rear, left,right
    '''


    (range_images, camera_projections,
     range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(
        frame)

    points, cp_points = frame_utils.convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose)
    points_ri2, cp_points_ri2 = frame_utils.convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose,
        ri_index=1)


    # 3d points in vehicle frame.
    # points_top = points[0]
    # points_front = points[1]
    # points_right = points[3]
    # points_left = points[2]
    # points_rear = points[4]



    points_all = np.concatenate(points, axis=0)
    points_all_ri2 = np.concatenate(points_ri2, axis=0)
    # camera projection corresponding to each point.
    cp_points_all = np.concatenate(cp_points, axis=0)
    cp_points_all_ri2 = np.concatenate(cp_points_ri2, axis=0)

    return points_all, points


#Adopted from waymo_utils PCDet generate_labels
def get_labels_from_frame(frame):

    def drop_info_with_name(info, name):
        ret_info = {}
        keep_indices = [i for i, x in enumerate(info['name']) if x != name]
        for key in info.keys():
            ret_info[key] = info[key][keep_indices]
        return ret_info

    WAYMO_CLASSES = ['unknown', 'Vehicle', 'Pedestrian', 'Sign', 'Cyclist']
    obj_name, difficulty, dimensions, locations, heading_angles = [], [], [], [], []
    tracking_difficulty, speeds, accelerations, obj_ids = [], [], [], []
    num_points_in_gt = []
    laser_labels = frame.laser_labels
    for i in range(len(laser_labels)):
        box = laser_labels[i].box
        class_ind = laser_labels[i].type
        loc = [box.center_x, box.center_y, box.center_z]
        heading_angles.append(box.heading)
        obj_name.append(WAYMO_CLASSES[class_ind])
        difficulty.append(laser_labels[i].detection_difficulty_level)
        tracking_difficulty.append(laser_labels[i].tracking_difficulty_level)
        dimensions.append([box.length, box.width, box.height])  # lwh in unified coordinate of OpenPCDet
        locations.append(loc)
        obj_ids.append(laser_labels[i].id)
        num_points_in_gt.append(laser_labels[i].num_lidar_points_in_box)

    annotations = {}
    annotations['name'] = np.array(obj_name)
    annotations['difficulty'] = np.array(difficulty)
    annotations['dimensions'] = np.array(dimensions)
    annotations['location'] = np.array(locations)
    annotations['heading_angles'] = np.array(heading_angles)

    annotations['obj_ids'] = np.array(obj_ids)
    annotations['tracking_difficulty'] = np.array(tracking_difficulty)
    annotations['num_points_in_gt'] = np.array(num_points_in_gt)

    annotations = drop_info_with_name(annotations, name='unknown')
    if annotations['name'].__len__() > 0:
        gt_boxes_lidar = np.concatenate([
            annotations['location'], annotations['dimensions'], annotations['heading_angles'][..., np.newaxis]],
            axis=1
        )
    else:
        gt_boxes_lidar = np.zeros((0, 7))
    annotations['gt_boxes_lidar'] = gt_boxes_lidar
    return annotations

def get_yaw_and_transformation(frame,lidar='top'):
    if lidar=='top':
        yaw = -0.55
        transform_matrix = np.reshape(np.array(frame.context.laser_calibrations[4].extrinsic.transform), [4, 4])
    elif lidar=='front':
        yaw = 0
        transform_matrix = np.reshape(np.array(frame.context.laser_calibrations[0].extrinsic.transform), [4, 4])
    elif lidar=='left':
        yaw = np.pi/2
        transform_matrix = np.reshape(np.array(frame.context.laser_calibrations[2].extrinsic.transform), [4, 4])
    elif lidar=='right':
        yaw = -np.pi/2
        transform_matrix = np.reshape(np.array(frame.context.laser_calibrations[3].extrinsic.transform), [4, 4])
    else: #rear
        yaw = -np.pi
        transform_matrix = np.reshape(np.array(frame.context.laser_calibrations[1].extrinsic.transform), [4, 4])

    return yaw, transform_matrix

def render_image_from_frame(frame):
    import matplotlib.patches as patches
    def show_camera_image(frame,camera_image, camera_labels, layout, cmap=None):
      """Show a camera image and the given camera labels."""

      ax = plt.subplot(*layout)

      # Draw the camera labels.
      for camera_labels in frame.camera_labels:
        # Ignore camera labels that do not correspond to this camera.
        if camera_labels.name != camera_image.name:
          continue

        # Iterate over the individual labels.
        for label in camera_labels.labels:
          # Draw the object bounding box.
          ax.add_patch(patches.Rectangle(
            xy=(label.box.center_x - 0.5 * label.box.length,
                label.box.center_y - 0.5 * label.box.width),
            width=label.box.length,
            height=label.box.width,
            linewidth=1,
            edgecolor='red',
            facecolor='none'))

      # Show the camera image.
      plt.imshow(tf.image.decode_jpeg(camera_image.image), cmap=cmap)
      plt.title(open_dataset.CameraName.Name.Name(camera_image.name))
      plt.grid(False)
      plt.axis('off')


    plt.figure(figsize=(40, 30))
    list_cam=[2,1,0,3,4]
    for index,cam in enumerate(list_cam):
        show_camera_image(frame,frame.images[cam], frame.camera_labels, [1, 5, index+1])


import matplotlib.patches as patches

def show_camera_image(frame,camera_image, camera_labels, layout, cmap=None):
  """Show a camera image and the given camera labels."""

  ax = plt.subplot(*layout)

  # Draw the camera labels.
  for camera_labels in frame.camera_labels:
    # Ignore camera labels that do not correspond to this camera.
    if camera_labels.name != camera_image.name:
      continue

    # Iterate over the individual labels.
    for label in camera_labels.labels:
      # Draw the object bounding box.
      ax.add_patch(patches.Rectangle(
        xy=(label.box.center_x - 0.5 * label.box.length,
            label.box.center_y - 0.5 * label.box.width),
        width=label.box.length,
        height=label.box.width,
        linewidth=2,
        edgecolor='orange',
        facecolor='none'))

  # Show the camera image.
  plt.imshow(tf.image.decode_jpeg(camera_image.image), cmap=cmap)
  plt.title(open_dataset.CameraName.Name.Name(camera_image.name))
  plt.grid(False)
  plt.axis('off')

def render_camera(frame,fig_size=(50,40),cam_name='all'):
    if cam_name == 'all':
        plt.figure(figsize=fig_size)
        list_cam=[2,1,0,3,4]
        for index,cam in enumerate(list_cam):
            show_camera_image(frame,frame.images[cam], frame.camera_labels, [1, 5, index+1])
        plt.show()
    else:
        cam_list = ['FRONT','FRONT_LEFT','SIDE_LEFT','FRONT_RIGHT','SIDE_RIGHT']
        cam = cam_list.index(cam_name)
        plt.figure(figsize=fig_size)
        show_camera_image(frame,frame.images[cam], frame.camera_labels, [1, 1, 1])
        plt.show()


#Visualisation

def render_sample_detection_3d(anno):
    bbox = anno["3d_boxes"]
    points = bbox[:,0:3]
    point_cloud = o3d.PointCloud()
    point_cloud.points = o3d.Vector3dVector(points)
    open3d.draw_geometries([point_cloud])


def render_sample_detection_matplot(anno):
    bbox = anno["3d_boxes"]
    points = bbox[:,0:3]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.scatter(points[:,0],points[:,1],points[:,2],label='lidar position',color='r')

def render_sample_detection_matplot2d(anno,marker='o'):
    bbox = anno["3d_boxes"]
    points = bbox[:,0:3]

    plt.xlabel("x axis")
    plt.ylabel("y axis")

    plt.scatter(0,0,c = 'r')
    plt.arrow(0,0,3,0)
    plt.arrow(0,0,0,3)
    plt.scatter(points[:,0],points[:,1],marker=marker)
    plt.axis('equal')
    plt.xlim([-80,80])
    plt.ylim([-80,80])
