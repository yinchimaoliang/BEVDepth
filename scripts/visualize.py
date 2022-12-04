import os
from argparse import ArgumentParser

import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import tensorflow as tf
from nuscenes.utils.data_classes import Box, LidarPointCloud
from pyquaternion import Quaternion
from waymo_open_dataset.protos import metrics_pb2

from bevdepth.datasets.nusc_det_dataset import \
    map_name_from_general_to_detection

WAYMO_CLASS_COLOR_MAP = [[255, 0, 0], [0, 0, 255], [0, 255, 0],
                         [128, 128, 128]]

LINE_IDXES_3D = [[0, 1], [1, 2], [2, 3], [0, 3], [0, 4], [1, 5], [2, 6],
                 [3, 7], [4, 5], [5, 6], [6, 7], [4, 7]]

WAYMO_FIG_IDXES = [1, 2, 3, 5, 7]


def parse_args():
    parser = ArgumentParser(add_help=False)
    parser.add_argument('dataset',
                        type=str,
                        help='Type of dataset',
                        choices=['nuscenes', 'waymo'])
    parser.add_argument('idx',
                        type=int,
                        help='Index of the dataset to be visualized.')
    parser.add_argument('result_path', help='Path of the result json file.')
    parser.add_argument('target_path',
                        help='Target path to save the visualization result.')

    args = parser.parse_args()
    return args


def get_ego_box(box_dict, ego2global_rotation, ego2global_translation):
    box = Box(
        box_dict['translation'],
        box_dict['size'],
        Quaternion(box_dict['rotation']),
    )
    trans = -np.array(ego2global_translation)
    rot = Quaternion(ego2global_rotation).inverse
    box.translate(trans)
    box.rotate(rot)
    box_xyz = np.array(box.center)
    box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
    box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
    box_velo = np.array(box.velocity[:2])
    return np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:
    """
    cosa = np.cos(angle)
    sina = np.sin(angle)
    zeros = np.zeros(points.shape[0])
    ones = np.ones(points.shape[0])
    rot_matrix = np.stack(
        (cosa, sina, zeros, -sina, cosa, zeros, zeros, zeros, ones),
        axis=1).reshape(-1, 3, 3)
    points_rot = np.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = np.concatenate((points_rot, points[:, :, 3:]), axis=-1)
    return points_rot


def get_corners(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading],
            (x, y, z) is the box center
    Returns:
    """
    template = (np.array((
        [1, 1, -1],
        [1, -1, -1],
        [-1, -1, -1],
        [-1, 1, -1],
        [1, 1, 1],
        [1, -1, 1],
        [-1, -1, 1],
        [-1, 1, 1],
    )) / 2)

    corners3d = np.tile(boxes3d[:, None, 3:6],
                        [1, 8, 1]) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.reshape(-1, 8, 3),
                                      boxes3d[:, 6]).reshape(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d


def get_bev_lines(corners):
    return [[[corners[i, 0], corners[(i + 1) % 4, 0]],
             [corners[i, 1], corners[(i + 1) % 4, 1]]] for i in range(4)]


def get_3d_lines(corners):
    ret = []
    for st, ed in [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7],
                   [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]:
        if corners[st, -1] > 0 and corners[ed, -1] > 0:
            ret.append([[corners[st, 0], corners[ed, 0]],
                        [corners[st, 1], corners[ed, 1]]])
    return ret


def get_cam_corners(corners, translation, rotation, cam_intrinsics):
    cam_corners = corners.copy()
    cam_corners -= np.array(translation)
    cam_corners = cam_corners @ Quaternion(rotation).inverse.rotation_matrix.T
    cam_corners = cam_corners @ np.array(cam_intrinsics).T
    valid = cam_corners[:, -1] > 0
    cam_corners /= cam_corners[:, 2:3]
    cam_corners[~valid] = 0
    return cam_corners


def draw_bbox_on_image(img, pred_bboxes, pred_classes, ego2sensor, intrin_mat):
    paddings = np.ones((pred_bboxes.shape[0], pred_bboxes.shape[1], 1),
                       dtype=np.float32)
    pred_bboxes = np.concatenate([pred_bboxes, paddings], axis=-1)
    pixel_coord_pred_bboxes = (
        intrin_mat @ ego2sensor
        @ pred_bboxes[..., np.newaxis])[:, :, :3].squeeze(-1)
    pixel_coord_pred_bboxes[..., :2] /= pixel_coord_pred_bboxes[..., 2:]
    for pixel_coord_pred_bbox, pred_class in zip(pixel_coord_pred_bboxes,
                                                 pred_classes):
        if np.any(pixel_coord_pred_bbox[:, -1] < 0):
            continue
        for line_idx_3d in LINE_IDXES_3D:
            img = cv2.line(img, [
                int(pixel_coord_pred_bbox[line_idx_3d[0]][0]),
                int(pixel_coord_pred_bbox[line_idx_3d[0]][1])
            ], [
                int(pixel_coord_pred_bbox[line_idx_3d[1]][0]),
                int(pixel_coord_pred_bbox[line_idx_3d[1]][1])
            ], WAYMO_CLASS_COLOR_MAP[pred_class])
    return img


def nuscenes_demo(
    idx,
    nusc_results_file,
    dump_file,
    threshold=0.0,
    show_range=60,
    show_classes=[
        'car',
        'truck',
        'construction_vehicle',
        'bus',
        'trailer',
        'barrier',
        'motorcycle',
        'bicycle',
        'pedestrian',
        'traffic_cone',
    ],
):
    # Set cameras
    IMG_KEYS = [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
        'CAM_BACK', 'CAM_BACK_LEFT'
    ]
    infos = mmcv.load('data/nuScenes/nuscenes_12hz_infos_val.pkl')
    assert idx < len(infos)
    # Get data from dataset
    results = mmcv.load(nusc_results_file)['results']
    info = infos[idx]
    lidar_path = info['lidar_infos']['LIDAR_TOP']['filename']
    lidar_points = np.fromfile(os.path.join('data/nuScenes', lidar_path),
                               dtype=np.float32,
                               count=-1).reshape(-1, 5)[..., :4]
    lidar_calibrated_sensor = info['lidar_infos']['LIDAR_TOP'][
        'calibrated_sensor']
    # Get point cloud
    pts = lidar_points.copy()
    ego2global_rotation = np.mean(
        [info['cam_infos'][cam]['ego_pose']['rotation'] for cam in IMG_KEYS],
        0)
    ego2global_translation = np.mean([
        info['cam_infos'][cam]['ego_pose']['translation'] for cam in IMG_KEYS
    ], 0)
    lidar_points = LidarPointCloud(lidar_points.T)
    lidar_points.rotate(
        Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix)
    lidar_points.translate(np.array(lidar_calibrated_sensor['translation']))
    pts = lidar_points.points.T

    # Get GT corners
    gt_corners = []
    for i in range(len(info['ann_infos'])):
        if map_name_from_general_to_detection[
                info['ann_infos'][i]['category_name']] in show_classes:
            box = get_ego_box(
                dict(
                    size=info['ann_infos'][i]['size'],
                    rotation=info['ann_infos'][i]['rotation'],
                    translation=info['ann_infos'][i]['translation'],
                ), ego2global_rotation, ego2global_translation)
            if np.linalg.norm(box[:2]) <= show_range:
                corners = get_corners(box[None])[0]
                gt_corners.append(corners)

    # Get prediction corners
    pred_corners, pred_class = [], []
    for box in results[info['sample_token']]:
        if box['detection_score'] >= threshold and box[
                'detection_name'] in show_classes:
            box3d = get_ego_box(box, ego2global_rotation,
                                ego2global_translation)
            box3d[2] += 0.5 * box3d[5]  # NOTE
            if np.linalg.norm(box3d[:2]) <= show_range:
                corners = get_corners(box3d[None])[0]
                pred_corners.append(corners)
                pred_class.append(box['detection_name'])

    # Set figure size
    plt.figure(figsize=(24, 8))

    for i, k in enumerate(IMG_KEYS):
        # Draw camera views
        fig_idx = i + 1 if i < 3 else i + 2
        plt.subplot(2, 4, fig_idx)

        # Set camera attributes
        plt.title(k)
        plt.axis('off')
        plt.xlim(0, 1600)
        plt.ylim(900, 0)

        img = mmcv.imread(
            os.path.join('data/nuScenes', info['cam_infos'][k]['filename']))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Draw images
        plt.imshow(img)

        # Draw 3D predictions
        for corners, cls in zip(pred_corners, pred_class):
            cam_corners = get_cam_corners(
                corners,
                info['cam_infos'][k]['calibrated_sensor']['translation'],
                info['cam_infos'][k]['calibrated_sensor']['rotation'],
                info['cam_infos'][k]['calibrated_sensor']['camera_intrinsic'])
            lines = get_3d_lines(cam_corners)
            for line in lines:
                plt.plot(line[0],
                         line[1],
                         c=cm.get_cmap('tab10')(show_classes.index(cls)))

    # Draw BEV
    plt.subplot(1, 4, 4)

    # Set BEV attributes
    plt.title('LIDAR_TOP')
    plt.axis('equal')
    plt.xlim(-40, 40)
    plt.ylim(-40, 40)

    # Draw point cloud
    plt.scatter(-pts[:, 1], pts[:, 0], s=0.01, c=pts[:, -1], cmap='gray')

    # Draw BEV GT boxes
    for corners in gt_corners:
        lines = get_bev_lines(corners)
        for line in lines:
            plt.plot([-x for x in line[1]],
                     line[0],
                     c='r',
                     label='ground truth')

    # Draw BEV predictions
    for corners in pred_corners:
        lines = get_bev_lines(corners)
        for line in lines:
            plt.plot([-x for x in line[1]], line[0], c='g', label='prediction')

    # Set legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(),
               by_label.keys(),
               loc='upper right',
               framealpha=1)

    # Save figure
    plt.tight_layout(w_pad=0, h_pad=2)
    plt.savefig(dump_file)


def waymo_demo(
    idx,
    waymo_results_file,
    dump_file,
    threshold=0.2,
    show_range=60,
    show_classes=['Vehicle', 'Pedestrian', 'Cyclist'],
):
    # Set cameras
    IMG_KEYS = [
        'FRONT_LEFT', 'FRONT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT'
    ]
    infos = mmcv.load('data/waymo/v1.4/waymo_infos_validation.pkl')
    gt_info = infos[idx]
    assert idx < len(infos)
    # Get data from dataset
    with tf.io.gfile.GFile(waymo_results_file, 'rb') as f:
        predictions_objects = metrics_pb2.Objects.FromString(f.read())
    pred_bboxes = list()
    pred_classes = list()
    pred_scores = list()
    for obj in predictions_objects.objects:
        if obj.frame_timestamp_micros == gt_info[
                'timestamp'] and obj.score > threshold:
            prediction_box = obj.object.box
            pred_bboxes.append(
                np.asarray([
                    prediction_box.center_x,
                    prediction_box.center_y,
                    prediction_box.center_z,
                    prediction_box.length,
                    prediction_box.width,
                    prediction_box.height,
                    prediction_box.heading,
                ]))
            pred_classes.append(obj.object.type)
            pred_scores.append(obj.score)
    lidar_points = np.fromfile(os.path.join(
        './data/waymo/v1.4', gt_info['lidar_infos']['lidar_points_path']),
                               dtype=np.float32,
                               count=-1).reshape(-1, 7)
    pred_corners = get_corners(np.stack(pred_bboxes))
    # Set figure size
    plt.figure(figsize=(24, 8))
    for i, img_key in enumerate(IMG_KEYS):
        fig_idx = WAYMO_FIG_IDXES[i]
        plt.subplot(2, 4, fig_idx)
        # Set camera attributes

        img = mmcv.imread(
            os.path.join('./data/waymo/v1.4',
                         gt_info['cam_infos'][img_key]['filename']))
        T_front_cam_to_ref = np.array(
            [[0.0, 0.0, 1.0, 0.0], [-1.0, -0.0, -0.0, -0.0],
             [-0.0, -1.0, -0.0, -0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float32)
        sensor2ego = gt_info['cam_infos'][img_key]['extrinsic'].reshape(
            4, 4) @ T_front_cam_to_ref
        intrin_mat = np.zeros((4, 4), dtype=np.float32)
        intrin_mat[3, 3] = 1
        intrin_mat[0, 0] = gt_info['cam_infos'][img_key]['intrinsic'][0]
        intrin_mat[1, 1] = gt_info['cam_infos'][img_key]['intrinsic'][1]
        intrin_mat[0, 2] = gt_info['cam_infos'][img_key]['intrinsic'][2]
        intrin_mat[1, 2] = gt_info['cam_infos'][img_key]['intrinsic'][3]
        intrin_mat[2, 2] = 1
        img_processed = draw_bbox_on_image(img, pred_corners, pred_classes,
                                           np.linalg.inv(sensor2ego),
                                           intrin_mat)
        mmcv.imwrite(img_processed, f'{img_key}.png')
        plt.title(img_key)
        plt.axis('off')
        plt.xlim(0, img_processed.shape[1])
        plt.ylim(img_processed.shape[0], 0)
        img_processed = cv2.cvtColor(img_processed, cv2.COLOR_BGR2RGB)
        plt.imshow(img_processed)
    # Draw BEV
    plt.subplot(1, 4, 4)

    # Set BEV attributes
    plt.title('LIDAR_TOP')
    plt.axis('equal')
    plt.xlim(-70, 70)
    plt.ylim(-30, 70)

    # Draw point cloud
    plt.scatter(-lidar_points[:, 1],
                lidar_points[:, 0],
                s=0.01,
                c=lidar_points[:, 3],
                cmap='gray')
    gt_bboxes = list()
    for gt_box3d, gt_class3d, gt_most_visible_camera_name in zip(
            gt_info['gt_boxes3d'], gt_info['gt_classes3d'],
            gt_info['gt_most_visible_camera_names']):
        if gt_most_visible_camera_name and gt_class3d in show_classes:
            gt_bboxes.append(gt_box3d)
    gt_bboxes = np.stack(gt_bboxes)
    gt_corners = get_corners(gt_bboxes)

    # Draw BEV GT boxes
    for corners in gt_corners:
        lines = get_bev_lines(corners)
        for line in lines:
            plt.plot([-x for x in line[1]],
                     line[0],
                     c='r',
                     label='ground truth')
    # Draw BEV predictions
    for corners in pred_corners:
        lines = get_bev_lines(corners)
        for line in lines:
            plt.plot([-x for x in line[1]], line[0], c='g', label='prediction')
    # Set legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(),
               by_label.keys(),
               loc='upper right',
               framealpha=1)

    # Save figure
    plt.tight_layout(w_pad=8, h_pad=2)
    plt.subplots_adjust(wspace=0.15, hspace=0)
    plt.savefig(dump_file)


if __name__ == '__main__':
    args = parse_args()
    if args.dataset == 'waymo':
        waymo_demo(
            args.idx,
            args.result_path,
            args.target_path,
        )
    elif args.dataset == 'nuscenes':
        nuscenes_demo(
            args.idx,
            args.result_path,
            args.target_path,
        )
