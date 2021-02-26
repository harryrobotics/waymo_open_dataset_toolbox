import open3d as o3d
import numpy as np


######################################################################
#####Try to eliminate this to get rid of pytorch######################
#####Borrow from PCDet################################################
import torch

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:
    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
#     print(rot_matrix)

    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot

#############################################################################

def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
    Returns:
        corners3d of N boxes    
    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]

    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)

    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d



def create_open3d_pc(lidar):
    # create open3d point cloud object
    pcd = o3.geometry.PointCloud()

    # assign points
    pcd.points = o3.utility.Vector3dVector(lidar['points'])

    return pcd

def pc_in_o3d(point_cloud):
    """
    Arg: points: (N, 3) - N: number of points
    Out: open3d point cloud object
    """
    #shape: number of points x 3
#     print(point_cloud.shape)
    pcd_object = o3d.geometry.PointCloud()
    pcd_object.points = o3d.utility.Vector3dVector(point_cloud)

    return pcd_object

def box_in_o3d(box_corners,color):
    """
    Arg: corners (8,3) -  1 box only
        color : list len 3 e.g [0,0,1]
    Out: open3d object
    """

    # update corresponding Open3d line set
#     color = [1, 0, 0]
    # point indices defining bounding box edges
    lines = [[0, 1], [1, 2], [2, 3], [3, 0],
             [0, 4], [1, 5], [2, 6], [3, 7],
             [4, 5], [5, 6], [6, 7], [7, 4]]
    colors = [color for _ in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(box_corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set

def multi_boxes_o3d(boxes_corners,color):
    """
    Arg: corners (N,8,3)
        color : list len 3 e.g [0,0,1]
    Out: open3d object for mutiple box
    """

    line_sets = []
    for corners in boxes_corners:
        boxo3d = box_in_o3d(corners,color)
        line_sets.append(boxo3d)

    return line_sets

def vis_pc_o3d(list_object):
    o3d.visualization.draw_geometries(list_object)

def vis_points(points,with_origin=False):
    """
    Arg: points: (N, 3) - N: number of points
    Out: open3d point cloud object
    """
    points_object = o3d.geometry.PointCloud()
    if with_origin == True:
        points = np.concatenate((points,np.array([[0,0,0]])))
    points_object.points = o3d.utility.Vector3dVector(points)

    o3d.visualization.draw_geometries([points_object])

def vis_points_with_box(bbox,points):
    """
    Arg: points: (N, 3) - N: number of points
        bbox: lwh
    Out: open3d point cloud object
    """
    box_corners = boxes_to_corners_3d(bbox.reshape(1,-1))

    box_o3d = box_in_o3d(box_corners.squeeze(0),color = [0,0,1])

    points_object = o3d.geometry.PointCloud()
    points_object.points = o3d.utility.Vector3dVector(points)

    o3d.visualization.draw_geometries([points_object,box_o3d])

def vis_points_with_multi_boxes(bboxes,points):
    """
    Arg: points: (N, 3) - N: number of points
        bbox: lwh
    Out: open3d point cloud object
    """
    box_corners = boxes_to_corners_3d(bboxes)

    boxes_o3d = multi_boxes_o3d(box_corners,color = [0,0,1])

    points_object = o3d.geometry.PointCloud()
    points_object.points = o3d.utility.Vector3dVector(points)

    boxes_o3d.append(points_object)

    o3d.visualization.draw_geometries(boxes_o3d)
