import numpy as np
import open3d as o3d
from tqdm import tqdm
import torch
import cv2
import os
import pickle
from copy import deepcopy
from pyquaternion import Quaternion
from collections import defaultdict
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from mmcv.ops import points_in_boxes_gpu, points_in_boxes_batch
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch.multiprocessing as mp

Cameralist = ['CAM_FRONT','CAM_FRONT_RIGHT','CAM_BACK_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_FRONT_LEFT']

nusc = NuScenes(version='v1.0-trainval', dataroot='./data/nuscenes', verbose=True)

target_map_list = ['singapore-onenorth','boston-seaport','singapore-queenstown','singapore-hollandvillage']
# target_map_list = ['singapore-queenstown']


def rotate_points_3d(
    points: np.ndarray,
    angle: float,
    axis: int = 2,
    clockwise: bool = True
) -> np.ndarray:

    assert points.shape[1] == 3, "points must have shape (N, 3)."
    assert axis in [0, 1, 2], "axis must be 0 (X), 1 (Y), or 2 (Z)."

    if clockwise:
        angle = -angle

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rotation_matrix = np.eye(3)

    if axis == 0: 
        rotation_matrix[1, 1] = cos_a
        rotation_matrix[1, 2] = -sin_a
        rotation_matrix[2, 1] = sin_a
        rotation_matrix[2, 2] = cos_a
    elif axis == 1:  
        rotation_matrix[0, 0] = cos_a
        rotation_matrix[0, 2] = sin_a
        rotation_matrix[2, 0] = -sin_a
        rotation_matrix[2, 2] = cos_a
    elif axis == 2:  
        rotation_matrix[0, 0] = cos_a
        rotation_matrix[0, 1] = -sin_a
        rotation_matrix[1, 0] = sin_a
        rotation_matrix[1, 1] = cos_a

    rotated_points = np.dot(points, rotation_matrix.T)

    return rotated_points


def process_scene(scene):

    first_sample_token = scene['first_sample_token']
    sample = nusc.get('sample', first_sample_token)
    local_target_list = {}
    local_box_list = {}

    while sample:

        lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_path = nusc.get_sample_data_path(lidar_data['token'])

        ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
        calibrated_sensor = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])

        lidar_to_car = transform_matrix(calibrated_sensor['translation'], Quaternion(calibrated_sensor['rotation']), inverse=False)
        car_to_global = transform_matrix(ego_pose['translation'], Quaternion(ego_pose['rotation']), inverse=False)
        global2car = transform_matrix(np.array(ego_pose['translation']),
                                         Quaternion(ego_pose['rotation']), inverse=True)

        lidar_points = LidarPointCloud.from_file(lidar_path).points[:3, :]

        points_homogeneous = np.vstack((lidar_points, np.ones((1, lidar_points.shape[1]))))
        points_ego = lidar_to_car @ points_homogeneous

        boxes = nusc.get_boxes(lidar_data['token'])

        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(-1, 1)
        dims[:,[0,1]] = dims[:,[1,0]]
        # gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
        gt_boxes = np.concatenate([locs, dims, rots], axis=1)

        locs_homogeneous = np.concatenate([gt_boxes[:, :3], np.ones((gt_boxes.shape[0], 1))], axis=1)
        ego_locs = (global2car @ locs_homogeneous.T).T[:, :3]
        ego_yaws = gt_boxes[:, 6] - Quaternion(ego_pose['rotation']).yaw_pitch_roll[0]

        gt_boxes = np.concatenate([ego_locs, gt_boxes[:, 3:6], ego_yaws.reshape(-1, 1)], axis=1)

        names = np.array([b.name for b in boxes])
        track_tokens = np.array([nusc.get('sample_annotation', b.token)['instance_token'] for b in boxes])

        valid_indices = [i for i, name in enumerate(names) if not name.startswith(('movable_object.', 'static_object.'))]
        gt_boxes = gt_boxes[valid_indices]
        track_tokens = track_tokens[valid_indices]
        names = names[valid_indices]

        clr_pts = []

        for cam in Cameralist:
            cam_data = nusc.get('sample_data', sample['data'][cam])
            cam_path = nusc.get_sample_data_path(cam_data['token'])
            img = cv2.imread(cam_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            calibrated_cam = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
            c2e_t = np.array(calibrated_cam['translation'])
            c2e_r = Quaternion(calibrated_cam['rotation'])
            car2cam = transform_matrix(c2e_t, c2e_r, inverse=True)
            I = np.array(calibrated_cam['camera_intrinsic'])

            points_camera = car2cam @ points_ego

            valid_mask = points_camera[2, :] > 0
            points_camera = points_camera[:, valid_mask]
            original_points = points_ego[:, valid_mask]

            points_2d = (I @ points_camera[:3, :]).T
            points_2d[:, 0] /= points_2d[:, 2]
            points_2d[:, 1] /= points_2d[:, 2]
            points_2d = points_2d[:, :2]

            H, W, _ = img.shape
            colored_points = []
            for i, (u, v) in enumerate(points_2d):
                u, v = int(round(u)), int(round(v))
                if 0 <= u < W and 0 <= v < H:
                    color = img[v, u]
                    colored_points.append(np.hstack((original_points[:3, i], color)))

            colored_points = np.array(colored_points)
            clr_pts.append(colored_points)

        clr_pts = np.vstack(clr_pts)
        # clr = clr_pts[:, 3:]
        # pts = np.hstack((clr_pts[:, :3], np.ones((clr_pts.shape[0], 1))))

        # points_global = car_to_global @ pts.T
        # points_global = np.concatenate([points_global[:3, :], clr.T], axis=0).T

        cut_box = deepcopy(gt_boxes)
        cut_box[:, 2] = cut_box[:, 2] - cut_box[:, 5] / 2
        cut_box[:, 4] = cut_box[:, 4] + 1
        cut_box[:, 3] = cut_box[:, 3] + 1
        cut_box[:, 5] = cut_box[:, 5] + 1

        clr_pts = torch.from_numpy(clr_pts).cuda().float()
        cut_box = torch.from_numpy(cut_box).cuda().float()

        pts_in_boxes = points_in_boxes_batch(clr_pts[:,:3].unsqueeze(0), cut_box.unsqueeze(0)).squeeze(0).cpu().numpy()

        clr_pts = clr_pts.cpu().numpy()
        cut_box = cut_box.cpu().numpy()

        for i, key in enumerate(track_tokens):
            box_indices = pts_in_boxes[:, i] != 0
            box_points = clr_pts[box_indices]
            if key in local_target_list:
                local_target_list[key].append(box_points.astype(np.float16))
                local_box_list[key].append(cut_box[i])
            else:
                local_target_list[key] = [box_points]
                local_box_list[key] = [cut_box[i]]

        sample = nusc.get('sample', sample['next']) if sample['next'] else None

    return local_target_list, local_box_list




def target_merge(tgt_root,box_root):

    tgt_instances = pickle.load(open(tgt_root, 'rb'))
    box_tgt = pickle.load(open(box_root, 'rb'))
    
    out_list = deepcopy(tgt_instances)
    for key, val in tgt_instances.items():
        pts_list = []
        for i, tgt_pts in enumerate(val):
            tgt_box = box_tgt[key][i]
            
            tgt_box[2] = tgt_box[2] + tgt_box[5] * 0.5
            tgt_pts[:,0:3] = tgt_pts[:,0:3] - tgt_box[0:3]
            tgt_pts[:,0:3] = rotate_points_3d(tgt_pts[:,0:3], tgt_box[6])
            pts_list.append(tgt_pts)

        if len(pts_list) > 0:
            pts_tracklets = np.concatenate(pts_list, axis=0)
            out_list[key] = pts_tracklets
        else:
            out_list[key] = np.empty((0, 6))

    save_root = r'./data/nuscenes/nuscenes_actor'
    os.makedirs(save_root, exist_ok=True)

    for key, val in out_list.items():
        print('saved',key,val.shape)
        np.save(f'{save_root}/{key}.npy',val)



def test():
    target_map = target_map_list[0]
    scenes_in_map = [scene for scene in nusc.scene if nusc.get('log', scene['log_token'])['location'] == target_map]
    scene = scenes_in_map[20]
    ltl,lbl = process_scene(scene)

if __name__ == "__main__":
    target_merge('nustarget_instances.pkl','nustarget_box.pkl')
    # test()
    # target_list = defaultdict(list)
    # box_list = defaultdict(list)
    # mp.set_start_method('spawn', force=True)

    # for target_map in target_map_list:
    #     scenes_in_map = [scene for scene in nusc.scene if nusc.get('log', scene['log_token'])['location'] == target_map]

    #     tasks = []
    #     with ProcessPoolExecutor(max_workers=8) as executor:
    #         for scene in scenes_in_map:
    #             tasks.append(executor.submit(process_scene, scene))

    #         for future in tqdm(as_completed(tasks), total=len(tasks), desc="Processing Scenes"):
    #             local_tgt,local_box = future.result()
    #             for key,val in local_tgt.items():
    #                 target_list[key].extend(val)
    #                 print("accumulate",key,len(target_list[key]))
    #             for key,val in local_box.items():
    #                 box_list[key].extend(val)

    # pickle.dump(target_list, open('target_instances2.pkl', 'wb'))
    # pickle.dump(box_list, open('target_box2.pkl', 'wb'))
