import numpy as np
import open3d as o3d
from tqdm import tqdm
import torch
import cv2
import os
from copy import deepcopy
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from mmcv.ops import points_in_boxes_gpu, points_in_boxes_batch
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch.multiprocessing as mp

Cameralist = ['CAM_FRONT','CAM_FRONT_RIGHT','CAM_BACK_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_FRONT_LEFT']

nusc = NuScenes(version='v1.0-trainval', dataroot='./data/nuscene', verbose=True)

target_map_list = ['singapore-onenorth','boston-seaport','singapore-queenstown','singapore-hollandvillage']
# target_map_list = ['singapore-onenorth']

import open3d
import numpy as np

def downsample_pts(pts_path,vox_size=0.01):
    pts = np.load(pts_path)

    pcd = open3d.geometry.PointCloud()  
    pcd.points = open3d.utility.Vector3dVector(pts[:,:3])
    pcd.colors = open3d.utility.Vector3dVector(pts[:,3:6] / 255)

    pcd = pcd.voxel_down_sample(voxel_size=vox_size)

    down_pts = np.hstack((np.asarray(pcd.points), np.asarray(pcd.colors) * 255)).astype(np.float32)

    return down_pts


def process_scene(scene):
    all_points = []
    first_sample_token = scene['first_sample_token']
    sample = nusc.get('sample', first_sample_token)

    while sample:
        lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_path = nusc.get_sample_data_path(lidar_data['token'])
        lidar_points = LidarPointCloud.from_file(lidar_path).points[:3, :]

        ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
        calibrated_sensor = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])

        lidar_to_car = transform_matrix(np.array(calibrated_sensor['translation']),
                                        Quaternion(calibrated_sensor['rotation']), inverse=False)
        car_to_global = transform_matrix(np.array(ego_pose['translation']),
                                         Quaternion(ego_pose['rotation']), inverse=False)

        points_homogeneous = np.vstack((lidar_points, np.ones((1, lidar_points.shape[1]))))
        points_ego = lidar_to_car @ points_homogeneous

        boxes = nusc.get_boxes(lidar_data['token'])
        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(-1, 1)
        gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
        gt_names = np.array([b.name for b in boxes])

        valid_indices = [i for i, name in enumerate(gt_names) 
                 if not name.startswith(('movable_object.', 'static_object.'))]

        gt_boxes = gt_boxes[valid_indices]  
        gt_names = gt_names[valid_indices]

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

        clr_pts = np.concatenate(clr_pts, axis=0)
        clr = clr_pts[:, 3:]
        pts = np.hstack((clr_pts[:, :3], np.ones((clr_pts.shape[0], 1))))

        points_global = car_to_global @ pts.T
        points_global = np.concatenate([points_global[:3, :], clr.T], axis=0).T

        cut_box = deepcopy(gt_boxes)
        cut_box[:, 2] = cut_box[:, 2] - cut_box[:, 5] / 2
        cut_box[:, 4] = cut_box[:, 4] + 1
        cut_box[:, 3] = cut_box[:, 3] + 1


        points_global = torch.from_numpy(points_global).cuda().float()
        cut_box = torch.from_numpy(cut_box).cuda().float()

        pts_in_boxes = points_in_boxes_gpu(points_global[:, :3].unsqueeze(0), cut_box.unsqueeze(0)).squeeze(0).cpu().numpy()
        points_global = points_global[pts_in_boxes.squeeze() == -1].cpu().numpy()

        all_points.append(points_global)

        sample = nusc.get('sample', sample['next']) if sample['next'] else None

    return np.vstack(all_points)


def test():
    target_map = target_map_list[0]
    scenes_in_map = [scene for scene in nusc.scene if nusc.get('log', scene['log_token'])['location'] == target_map]
    scene_list = [scenes_in_map[50], scenes_in_map[51]]
    pts = []
    for scene in scene_list:
        final_point_cloud = process_scene(scene)
        pts.append(final_point_cloud)
    final_point_cloud = np.vstack(pts)

if __name__ == "__main__":
    # test()
    mp.set_start_method('spawn', force=True)

    for target_map in target_map_list:
        scenes_in_map = [scene for scene in nusc.scene if nusc.get('log', scene['log_token'])['location'] == target_map]
        final_point_clouds = []
        tasks = []
        with ProcessPoolExecutor(max_workers=8) as executor:
            for scene in scenes_in_map:
                tasks.append(executor.submit(process_scene, scene))

            for future in tqdm(as_completed(tasks), total=len(tasks), desc="Processing Scenes"):
                result = future.result()
                final_point_clouds.append(result)

        final_point_cloud = np.vstack(final_point_clouds)
        print(f"Final Point Cloud Shape: {final_point_cloud.shape}")

        np.save(f"./data/{target_map}_point_cloud2.npy", final_point_cloud)
    
    # for target_map in target_map_list:
    #     down_pts = downsample_pts(f"./data/{target_map}_point_cloud.npy",vox_size=0.01)
    #     np.save(f"./data/{target_map}_pc_down.npy", down_pts)
    #     print(f"Final Point Cloud Shape: {down_pts.shape}")

    # down_pts = np.load(f"./data/boston-seaport_pc_down.npy")
    # print(f"Shape: {down_pts.dtype}")
    # pts = np.load("./data/boston-seaport_point_cloud.npy")
    # print(f"Shape: {pts.dtype}")
