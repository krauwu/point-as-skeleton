import glob
import os
import open3d
import sys
from tqdm import tqdm
import torch
import numpy as np
import cv2
import pickle
from collections import defaultdict


from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch.multiprocessing as mp


from nuplan.database.nuplan_db_orm.frame import Frame
from nuplan.database.utils.label.utils import raw_mapping
from nuplan.database.nuplan_db_orm.nuplandb_wrapper import NuPlanDBWrapper
from nuplan.database.nuplan_db.nuplan_scenario_queries import (get_images_from_lidar_tokens, get_ego_state_for_lidarpc_token_from_db)
from nuplan.database.nuplan_db_orm.lidar_box import LidarBox

from mmcv.ops import points_in_boxes_gpu, points_in_boxes_batch

NUPLAN_DATA_ROOT = os.getenv('NUPLAN_DATA_ROOT', './data/nuplan/nuplan-v1.1')
NUPLAN_MAPS_ROOT = os.getenv('NUPLAN_MAPS_ROOT', './data/nuplan/nuplan-v1.1/maps')
NUPLAN_DB_FILES = os.getenv('NUPLAN_DB_FILES', './data/nuplan/nuplan-v1.1/splits/mini')
NUPLAN_MAP_VERSION = os.getenv('NUPLAN_MAP_VERSION', 'nuplan-maps-v1.0')
BLOB_PATH = os.getenv('BLOB_PATH', './data/nuplan/nuplan-v1.1/sensor_blobs')


MAP_NAME = "us-nv-las-vegas"

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]


nuplandb_wrapper = NuPlanDBWrapper(
    data_root=NUPLAN_DATA_ROOT,
    map_root=NUPLAN_MAPS_ROOT,
    db_files=NUPLAN_DB_FILES,
    map_version=NUPLAN_MAP_VERSION,
)

# ,'CAM_R0'
gt_list = ['car','ped','bike']
Cameralist = ['CAM_F0','CAM_L0','CAM_L1','CAM_L2','CAM_B0','CAM_R2','CAM_R1']

log_db_names = ['2021.05.12.22.28.35_veh-35_00620_01164']


instance_root = './data/nuplan/actors/target_instances_m.pkl'
box_root = './data/nuplan/actors/target_box_m.pkl'

def enlarged_box(bbox, extra_width):
    """Enlarge the length, width and height boxes.

    Args:
        extra_width (float | torch.Tensor): Extra width to enlarge the box.

    Returns:
        :obj:`LiDARInstance3DBoxes`: Enlarged boxes.
    """
    enlarged_boxes = bbox
    enlarged_boxes[:, 6] += extra_width * 2
    # bottom center z minus extra_width
    enlarged_boxes[:, 2] -= extra_width
    return enlarged_boxes

def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d

def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        vis.add_geometry(line_set)

        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis

def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.ones(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if points.shape[1] > 3:
        pts.colors = open3d.utility.Vector3dVector(points[:, 3:])
    elif point_colors is not None:
        pts.colors = open3d.utility.Vector3dVector(point_colors)
    else:
        pts.colors = open3d.utility.Vector3dVector(np.zeros((points.shape[0], 3)))

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0, 0, 1))

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)

    vis.run()
    vis.destroy_window()

def downsample_pts(pts):
    ply = open3d.geometry.PointCloud()
    ply.points = open3d.utility.Vector3dVector(pts[:, :3])
    ply.colors = open3d.utility.Vector3dVector(pts[:, 3:])
    ply = ply.voxel_down_sample(voxel_size=0.001)
    
    # ply, _ = ply.remove_radius_outlier(nb_points=10, radius=0.05)
    pts = np.asarray(ply.points)
    clr = np.asarray(ply.colors)
    pts = np.concatenate([pts,clr],axis=1)
    print('downsampled',pts.shape)

    return pts

def target_extract(lidar_root, db_name, idx):

    db_record = nuplandb_wrapper.get_log_db(db_name)
    local_target_list = {}
    local_box_list = {}

    scene_token = db_record.lidar_pc[idx].token
    
    lid_record = db_record.lidar[0] 
    l2e_t = lid_record.translation_np
    l2e_r = lid_record.quaternion.rotation_matrix

    colored_pts = []
    
    # get pts and gtbox
    lidar_file = f'{lidar_root}/{scene_token}.pcd'
    boxes = db_record.lidar_pc[idx].boxes(frame=Frame.SENSOR)
    lidar_boxes = db_record.session.query(LidarBox).filter(LidarBox.lidar_pc_token == scene_token).all()
    track_token = np.array([box.track_token for box in lidar_boxes])

    locs = np.array([b.center for b in boxes]).reshape(-1, 3)
    dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
    dims[:,[0,1]] = dims[:,[1,0]]
    rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(-1, 1)
    labels = np.array([b.label for b in boxes])
    names = np.array([raw_mapping["id2local"][l] for l in labels])
    gt_boxes = np.concatenate([locs, dims, rots], axis=1)

    pcd = open3d.io.read_point_cloud(lidar_file)
    points = np.asarray(pcd.points)
    
    for cam in Cameralist:
        img_info = list(get_images_from_lidar_tokens(os.path.join(NUPLAN_DB_FILES, log_db_names[0]+".db"), [scene_token], [cam]))
        cam_db = db_record.camera.select_one(channel = cam)
        if len(img_info) != 0:
            img_path = os.path.join(BLOB_PATH, 'nuplan-v1.1_mini_camera_0', img_info[0].filename_jpg)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            c2e_t = cam_db.translation_np
            c2e_r = cam_db.quaternion.rotation_matrix
            inv_c2e_r = np.linalg.inv(c2e_r)
            I = np.float64(cam_db.intrinsic)
            distortion = np.float64(cam_db.distortion)
            
            points_camera = points @ (inv_c2e_r @ l2e_r).T + (l2e_t @ inv_c2e_r.T) - (c2e_t @ inv_c2e_r.T)

            valid_mask = points_camera[:, 2] > 0
            points_camera = points_camera[valid_mask]
            original_points = points[valid_mask]

            points_2d = (I @ points_camera.T).T  # (N, 3)
            points_2d[:, 0] /= points_2d[:, 2]  # u = fx * x / z + cx
            points_2d[:, 1] /= points_2d[:, 2]  # v = fy * y / z + cy
            points_2d = points_2d[:, :2]

            points_2d = cv2.undistortPoints(points_2d.reshape(-1, 1, 2), I, distortion, None, I).reshape(-1, 2)

            H, W, _ = img.shape
            colors = []
            valid_points = []

            for i, (u, v) in enumerate(points_2d):
                u, v = int(round(u)), int(round(v))
                if 0 <= u < W and 0 <= v < H:
                    colors.append(img[v, u])
                    valid_points.append(original_points[i])
            
            valid_points = np.array(valid_points)  # (M, 3)
            colors = np.array(colors)  # (M, 3)
            colored_pointcloud = np.hstack((valid_points, colors))  # (M, 6)
            colored_pts.append(colored_pointcloud)

    colored_pts = np.concatenate(colored_pts, axis=0)

    mask = np.where(labels!=0, True, False)
    gt_boxes = gt_boxes[mask]
    track_token = track_token[mask]
    names = names[mask]

    mask_gt = np.where(np.isin(names,gt_list), True, False)
    gt_boxes = gt_boxes[mask_gt]
    track_token = track_token[mask_gt]
    names = names[mask_gt]

    cut_box = deepcopy(gt_boxes)
    cut_box[:, 2] = cut_box[:, 2] - cut_box[:, 5] / 2

    mask = cut_box[:, 5]<3
    cut_box = cut_box[mask]
    gt_boxes = gt_boxes[mask]
    names = names[mask]
    track_token = track_token[mask]

    
    colored_pts = torch.from_numpy(colored_pts).cuda().float()
    cut_box = torch.from_numpy(cut_box).cuda().float()
    pts_in_boxes = points_in_boxes_batch(colored_pts[:,:3].unsqueeze(0), cut_box.unsqueeze(0)).squeeze(0).cpu().numpy()
    colored_pts = colored_pts.cpu().numpy()
    cut_box = cut_box.cpu().numpy()
    cut_box[:,[3,4]] = cut_box[:,[4,3]]
    for i, key in enumerate(track_token):
        box_indices = pts_in_boxes[:, i] != 0
        box_points = colored_pts[box_indices]
        if key in local_target_list:
            local_target_list[key].append(box_points)
            local_box_list[key].append(cut_box[i])
        else:
            local_target_list[key] = [box_points]
            local_box_list[key] = [cut_box[i]]
    return local_target_list,local_box_list
        
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

def target_merge(tgt_root,box_root):

    tgt_instances = pickle.load(open(tgt_root, 'rb'))
    box_tgt = pickle.load(open(box_root, 'rb'))
    
    out_list = deepcopy(tgt_instances)
    for key, val in tgt_instances.items():
        pts_list = []
        for i, tgt_pts in enumerate(val):
            tgt_box = box_tgt[key][i]

            if tgt_pts.shape[0] < 10:
                continue
            
            tgt_box[2] = tgt_box[2] + tgt_box[5] * 0.5
            tgt_pts[:,0:3] = tgt_pts[:,0:3] - tgt_box[0:3]
            tgt_pts[:,0:3] = rotate_points_3d(tgt_pts[:,0:3], tgt_box[6])
            pts_list.append(tgt_pts)

        if len(pts_list) > 0:
            pts_tracklets = np.concatenate(pts_list, axis=0)
            # pts = pts_tracklets[:,0:3]
            # clr = pts_tracklets[:,3:6] / 255
            # pts_draw = np.concatenate([pts,clr],axis=1)
            # # downsample_pts(pts_tracklets)
            # draw_scenes(pts_draw, gt_boxes=np.array([tgt_box*[0,0,0,1,1,1,0]]), draw_origin=True)
            out_list[key] = pts_tracklets
        else:
            out_list[key] = np.empty((0, 6))

    save_root = './data/nuplan/actors/track_actor'
    os.makedirs(save_root, exist_ok=True)

    for key, val in out_list.items():
        print('saved',key,val.shape)
        if val.shape[0] > 1000000:
            val = downsample_pts(val)
            # pts = val[:,0:3]
            # clr = val[:,3:6] / 255
            # pts_draw = np.concatenate([pts,clr],axis=1)
            # draw_scenes(pts_draw, draw_origin=True)
        np.save(f'{save_root}/{key}.npy',val)


if __name__ == "__main__":
    # cam_z_angle()
    # downsample_pts(np.load('D:/CODE/simgen/pts.npy'))
    # read_pcds(log_db_names)
    # li2global_color(lidar_file,db_records,100)
    target_merge(instance_root, box_root)
    
    # import multiprocessing
    # multiprocessing.set_start_method("spawn", force=True) 
    # mp.set_start_method('spawn', force=True)
     
    # target_list = defaultdict(list)
    # box_list = defaultdict(list)
    
    # for db_name in log_db_names:
        
    #     lidar_file = f'./data/nuplan/nuplan-v1.1/sensor_blobs/nuplan-v1.1_mini_lidar_0/{db_name}/MergedPointCloud'
        
    #     pcds = glob.glob(lidar_file + '/*.pcd')
    #     tasks = []
        
    #     internal = list(range(0, len(pcds), 5))
    #     # internal = list(range(0, 10, 2))
    #     with ProcessPoolExecutor(max_workers=4) as executor:
    #         for idx in internal:
    #             tasks.append(executor.submit(target_extract, lidar_file, db_name, idx))

    #         for future in  tqdm(as_completed(tasks), total=len(tasks), desc="Processing Scenes"):
    #             for key,val in local_tgt.items():
    #                 target_list[key].extend(val)
    #                 print("accumulate",key,len(target_list[key]))
    #             for key,val in local_box.items():
    #                 box_list[key].extend(val)

    # pickle.dump(target_list, open('target_instances.pkl', 'wb'))
    # pickle.dump(box_list, open('target_box.pkl', 'wb'))
