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


def create_simplified_infos(infos,class_names):

    gt_annos = []
    for info in infos:
        #metadata
        gt_lidar_sequence = info['point_cloud']['lidar_sequence']
        gt_frame_id = info['frame_id']
        gt_sample_index = info['point_cloud']['sample_idx']

        gt_names = info['annos']['name']

        labels = []
        for name in gt_names:
            index = class_names.index(name)
            labels.append(index)
        labels = np.asarray(labels,dtype='int')

        gt_boxes = info['annos']['gt_boxes_lidar']

        object_id = info['annos']['obj_ids']


        gt_annos.append({
            "name":
            gt_names,
            "gt_labels":
            labels,
            "3d_boxes":
            gt_boxes,
            "object_id":
            object_id,
            "metadata":{"lidar_sequence": gt_lidar_sequence,
                        "frame_id": gt_frame_id,
                        "sample_idx": gt_sample_index}
        })
    return gt_annos

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
    #0 top 1 front 2 left 3 right 4 rear
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
#Search

def search_sequence_from_info(sequence_name,infos):
    '''
    Args: sequence_name
    infos dict
    Output: Start end index of sequence in info dict
    '''
    start_index = None
    stop_index = None
    for index,info in enumerate(infos):
        # print(info['point_cloud']['lidar_sequence'])
        # print(sequence_name)
        if info['point_cloud']['lidar_sequence'] == sequence_name:
            if start_index == None:
                start_index = index
    stop_index = index
    if start_index == None:
        return None, None
    return start_index,stop_index



def get_frame_from_index(segment,frame_index):
    for index,data in enumerate(segment):
        if index ==frame_index:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            return frame
    return None


#Filter functions

def filter_annos_class(annos, used_class):
    new_annos = []
    for anno in annos:
        filtered_annotations = {}
        relevant_annotation_indices = [
            i for i, x in enumerate(anno['name']) if x in used_class
        ]
#         print(relevant_annotation_indices)
        for key in anno.keys():
            if (key in ['metadata','frame_id','token']):
                filtered_annotations[key] = anno[key]
            else:
                filtered_annotations[key] = (
                    anno[key][relevant_annotation_indices])
        new_annos.append(filtered_annotations)
    return new_annos


def filter_annos_front_vehicle(annos,theshold = 0.4):
    new_annos = []
    for anno in annos:
        filtered_annotations = {}
        relevant_annotation_indices = [i for i, x in enumerate(anno['rotation'])
                                        if ( (anno['relative_angle'][i] - x) > - theshold
                                        and (anno['relative_angle'][i] - x) < theshold)]
        for key in anno.keys():
            if (key in ['metadata','frame_id','token']):
                filtered_annotations[key] = anno[key]
            else:
                filtered_annotations[key] = (
                    anno[key][relevant_annotation_indices])
        new_annos.append(filtered_annotations)
    return new_annos

def filter_annos_by_rotation(annos,angle):
    new_annos = []
    for anno in annos:
        filtered_annotations = {}
        relevant_annotation_indices = [i for i, x in enumerate(anno['3d_boxes'][:,6]) if ( x < angle )]
        for key in anno.keys():
            if (key in ['metadata','frame_id','token']):
                filtered_annotations[key] = anno[key]
            else:
                filtered_annotations[key] = (anno[key][relevant_annotation_indices])
        new_annos.append(filtered_annotations)
    return new_annos

def filter_true_positive(annos,selected_list):
    new_image_annos = []
    if isinstance(annos, dict):
        img_filtered_annotations = {}
        relevant_annotation_indices = selected_list
#         print(relevant_annotation_indices)
#         print(annos)
        for key in annos.keys():
            if (key in ['metadata','frame_id','token']):
                img_filtered_annotations[key] = annos[key]
            else:
#                 print(key)
                img_filtered_annotations[key] = (
                    annos[key][relevant_annotation_indices])
        new_image_annos.append(img_filtered_annotations)

    else:
        for j, anno in enumerate(annos):
#             print(j)
            img_filtered_annotations = {}
            relevant_annotation_indices = selected_list[j]
#             print(relevant_annotation_indices)
#             print(anno)
            for key in anno.keys():
                if (key in ['metadata','frame_id','token']):
                    img_filtered_annotations[key] = anno[key]
                else:

                    img_filtered_annotations[key] = (
                        anno[key][relevant_annotation_indices])
            new_image_annos.append(img_filtered_annotations)
    return new_image_annos



def distance(anno):
    new_anno =anno.copy()
    new_anno['distance'] = np.array([])
    for i,pos in enumerate(new_anno['3d_boxes']):
        distance = np.linalg.norm(pos[0:3])
        new_anno['distance']=np.append(new_anno['distance'],distance)
    return new_anno

def relative_angle(anno):
    new_anno = anno.copy()
    new_anno['relative_angle'] = np.array([])
    for i,pos in enumerate(new_anno['3d_boxes']):
        angle = np.arctan2(pos[1],pos[0]) #return angle from 0 to pi or 0 to -pi
        new_anno['relative_angle']=np.append(new_anno['relative_angle'],angle)
    return new_anno

def rotation_correction(anno): #convert the rotation y to the same coordinate with relative angle
    new_anno = anno.copy()
    new_anno['rotation'] = np.array([])
    for i,angle in enumerate(new_anno['3d_boxes'][:,6]):
        # angle = -angle - np.pi/2
#         if (angle > np.pi):
#             angle = angle - np.pi
#         if (angle < -np.pi):
#             angle = np.pi + angle
        new_anno['rotation'] = np.append(new_anno['rotation'],angle)
    return new_anno


def add_distance(annos):
    distance_anno = annos.copy()
    for i in range(len(annos)):
        distance_anno[i] = distance(annos[i])
    return distance_anno

def add_relative_angle(annos):
    angle_anno = annos.copy()
    for i in range(len(annos)):
        angle_anno[i] = relative_angle(annos[i])
    return angle_anno

def add_rotation_correction(annos):
    angle_anno = annos.copy()
    for i in range(len(annos)):
        angle_anno[i] = rotation_correction(annos[i])
    return angle_anno














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
