import glob
import os
import open3d
import sys
from tqdm import tqdm
import torch
import numpy as np
import cv2
import pickle
import yaml

from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed

from nuplan.database.nuplan_db_orm.frame import Frame
from nuplan.database.utils.label.utils import raw_mapping
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario, CameraChannel, LidarChannel
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioExtractionInfo
from nuplan.database.nuplan_db.nuplan_scenario_queries import get_images_from_lidar_tokens
from nuplan.database.nuplan_db_orm.nuplandb_wrapper import NuPlanDBWrapper

from mmcv.ops import points_in_boxes_gpu, points_in_boxes_batch


NUPLAN_DATA_ROOT = os.getenv('NUPLAN_DATA_ROOT', 'D:/data/nuplan')
NUPLAN_MAPS_ROOT = os.getenv('NUPLAN_MAPS_ROOT', 'D:/data/nuplan/nuplan-v1.1/maps')
NUPLAN_DB_FILES = os.getenv('NUPLAN_DB_FILES', 'D:/data/nuplan/nuplan-v1.1/splits/mini')
NUPLAN_MAP_VERSION = os.getenv('NUPLAN_MAP_VERSION', 'nuplan-maps-v1.0')
BLOB_PATH = os.getenv('BLOB_PATH', 'D:/data/nuplan/nuplan-v1.1/sensor_blobs')

TEST_DB_FILE = "d:/data/nuplan/nuplan-v1.1/splits/mini/2021.05.12.22.00.38_veh-35_01008_01518.db"
MAP_NAME = "us-nv-las-vegas"

TEST_INITIAL_LIDAR_PC = "58ccd3df9eab54a3"
TEST_INITIAL_TIMESTAMP = 1620858198150622

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

log_db_name = "2021.05.12.22.00.38_veh-35_01008_01518"
lidar_file = f'D:/data/nuplan/nuplan-v1.1/sensor_blobs/nuplan-v1.1_mini_lidar_0/{log_db_name}/MergedPointCloud'
img_file = f'D:/data/nuplan/nuplan-v1.1/sensor_blobs/nuplan-v1.1_mini_camera_0/{log_db_name}'
db_records = nuplandb_wrapper.get_log_db(log_db_name)

instance_root = 'D:/CODE/simgen/target_instances.pkl'
box_root = 'D:/CODE/simgen/target_box.pkl'

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
        pts.colors = open3d.utility.Vector3dVector(points[:, 3:]/255)
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


def read_pcds(path_file):
    pcds = glob.glob(path_file + '/*.pcd')

    tgt_box = dict({
        'car': [],
        'ped': [],
        'bike': []
    })

    tgt_instance = dict({
        'car': [],
        'ped': [],
        'bike': []
    })

    pts_list = []
    tasks = []
    internal = list(range(0, len(pcds), 10))
    with ProcessPoolExecutor(max_workers=8) as executor:
        for idx in internal:
            tasks.append(executor.submit(target_extract, lidar_file, db_records, idx))

        for future in tqdm(as_completed(tasks), total=len(tasks), desc=f'Processing frame'):
            partial_tgt_ins, partial_tgt_box = future.result()
            for key in tgt_box:
                tgt_box[key].extend(partial_tgt_box[key])
                tgt_instance[key].extend(partial_tgt_ins[key])
            print(len(tgt_box['car']), len(tgt_box['ped']), len(tgt_box['bike']))

    pickle.dump(tgt_instance, open('target_instances.pkl', 'wb'))
    pickle.dump(tgt_box, open('target_box.pkl', 'wb'))

    # for idx in tqdm(range(len(pcds)), desc=f'Processing frame'):
    #     if count == 10:
    #         pts = li2global(lidar_file,db_records,idx)
    #         pts_list.append(pts)
    #         count = 0
    #     else:
    #         count += 1
    # pts = np.concatenate(pts_list,axis=0)
    # ply = open3d.geometry.PointCloud()
    # ply.points = open3d.utility.Vector3dVector(pts[:, :3])
    # ply.colors = open3d.utility.Vector3dVector(pts[:, 3:] / 255)


    # return pts

def downsample_pts_sedan(pts):
    ply = open3d.geometry.PointCloud()
    ply.points = open3d.utility.Vector3dVector(pts[:, :3])
    
    # ply = ply.voxel_down_sample(voxel_size=0.01)
    ply = ply.uniform_down_sample(40)
    ply = ply.voxel_down_sample(voxel_size=0.01)
    
    ply, _ = ply.remove_radius_outlier(nb_points=10, radius=0.05)

    pts = np.asarray(ply.points)

    return pts

def downsample_pts_suv(pts):
    ply = open3d.geometry.PointCloud()
    ply.points = open3d.utility.Vector3dVector(pts[:, :3])
    
    # ply = ply.voxel_down_sample(voxel_size=0.01)
    ply = ply.uniform_down_sample(15)
    # ply = ply.voxel_down_sample(voxel_size=0.01)
    
    ply, _ = ply.remove_radius_outlier(nb_points=10, radius=0.05)

    pts = np.asarray(ply.points)

    return pts

def downsample_pts_van(pts):
    ply = open3d.geometry.PointCloud()
    ply.points = open3d.utility.Vector3dVector(pts[:, :3])
    
    # ply = ply.voxel_down_sample(voxel_size=0.01)
    ply = ply.uniform_down_sample(15)
    # ply = ply.voxel_down_sample(voxel_size=0.01)
    
    ply, _ = ply.remove_radius_outlier(nb_points=4, radius=0.05)

    pts = np.asarray(ply.points)

    return pts

def downsample_pts_ped(pts):
    ply = open3d.geometry.PointCloud()
    ply.points = open3d.utility.Vector3dVector(pts[:, :3])
    
    ply, _ = ply.remove_radius_outlier(nb_points=4, radius=0.06)
    ply = ply.uniform_down_sample(3)

    pts = np.asarray(ply.points)

    return pts

def cam_z_angle():
    cam_db_l2 = db_records.camera.select_one(channel = 'CAM_L1')
    cam_db_r2 = db_records.camera.select_one(channel = 'CAM_R1')
    
    # c2e_t_l2 = cam_db_l2.translation_np
    # c2e_t_r2 = cam_db_r2.translation_np
    c2e_r_l2 = cam_db_l2.quaternion.rotation_matrix
    c2e_r_r2 = cam_db_r2.quaternion.rotation_matrix

    z_l2 = c2e_r_l2[:, 2]
    z_r2 = c2e_r_r2[:, 2]

    dot_product = np.dot(z_l2, z_r2)

    norm_z_l2 = np.linalg.norm(z_l2)
    norm_z_r2 = np.linalg.norm(z_r2)

    cos_theta = dot_product / (norm_z_l2 * norm_z_r2)

    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    theta_degrees = np.degrees(theta)

    print(f"Camera z-axis angle (rad): {theta}")
    print(f"Camera z-axis angle (deg): {theta_degrees}")

def target_extract(lidar_root, db_record, idx):

    local_tgt_box = {
        'car': [],
        'ped': [],
        'bike': []
    }

    local_tgt_instances = {
        'car': [],
        'ped': [],
        'bike': []
    }

    scene_token = db_record.lidar_pc[idx].token

    # get pts and gtbox
    lidar_file = f'{lidar_root}/{scene_token}.pcd'
    boxes = db_record.lidar_pc[idx].boxes(frame=Frame.SENSOR)

    locs = np.array([b.center for b in boxes]).reshape(-1, 3)
    dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
    dims[:,[0,1]] = dims[:,[1,0]]
    rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(-1, 1)
    labels = np.array([b.label for b in boxes])
    names = np.array([raw_mapping["id2local"][l] for l in labels])
    gt_boxes = np.concatenate([locs, dims, rots], axis=1)

    pcd = open3d.io.read_point_cloud(lidar_file)
    points = np.asarray(pcd.points)

    mask = np.where(labels!=0, True, False)
    gt_boxes = gt_boxes[mask]
    names = names[mask]
    mask_gt = np.where(np.isin(names,gt_list), True, False)
    names = names[mask_gt]
    gt_boxes = gt_boxes[mask_gt]

    cut_box = deepcopy(gt_boxes)
    cut_box[:, 2] = cut_box[:, 2] - cut_box[:, 5] / 2

    mask = cut_box[:, 5]<3
    cut_box = cut_box[mask]
    gt_boxes = gt_boxes[mask]
    names = names[mask]

    points = torch.from_numpy(points).cuda().float()
    cut_box = torch.from_numpy(cut_box).cuda().float()

    pts_in_boxes = points_in_boxes_batch(points.unsqueeze(0), cut_box.unsqueeze(0)).squeeze(0).cpu().numpy()

    points = points.cpu().numpy()
    cut_box = cut_box.cpu().numpy()

    for i in range(cut_box.shape[0]):
        box_indices = pts_in_boxes[:, i] != 0
        box_points = points[box_indices]
        local_tgt_instances[names[i]].append(box_points)
        local_tgt_box[names[i]].append(gt_boxes[i])

    return local_tgt_instances, local_tgt_box
        
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


def target_merge(tgt_root, tgt_box_root):

    tgt_instances = pickle.load(open(tgt_root, 'rb'))
    tgt_boxes = pickle.load(open(tgt_box_root, 'rb'))

    sedan = []
    suv = []
    pickup = []
    
    ped =  []

    bike = []
    
    for i in range(len(tgt_instances['car'])):
        tgt_pts = tgt_instances['car'][i]
        tgt_box = tgt_boxes['car'][i]

        tgt_pts = tgt_pts - tgt_box[0:3]
        tgt_pts = rotate_points_3d(tgt_pts, tgt_box[6])
        tgt_box[0:3] = 0; tgt_box[6] = 0
        if tgt_box[3] < 4 and tgt_box[5] > 2.9:
            continue
        elif tgt_box[3] < 6 and tgt_box[5] < 1.8 and tgt_pts.shape[0] > 100:
            sedan.append(tgt_pts)
        elif tgt_box[5] < 2.05 and tgt_pts.shape[0] > 20:
            suv.append(tgt_pts)
        elif tgt_box[5] > 2.1:
            pickup.append(tgt_pts)

    for i in range(len(tgt_instances['ped'])):
        tgt_pts = tgt_instances['ped'][i]
        tgt_box = tgt_boxes['ped'][i]

        tgt_pts = tgt_pts - tgt_box[0:3]
        tgt_pts = rotate_points_3d(tgt_pts, tgt_box[6])
        tgt_box[0:3] = 0; tgt_box[6] = 0
        if tgt_pts.shape[0] < 900 and tgt_pts.shape[0] > 800 and tgt_box[3] < 1 and tgt_box[4] < 1:
            ped.append(tgt_pts)

    for i in range(len(tgt_instances['bike'])):
        tgt_pts = tgt_instances['bike'][i]
        tgt_box = tgt_boxes['bike'][i]

        tgt_pts = tgt_pts - tgt_box[0:3]
        tgt_pts = rotate_points_3d(tgt_pts, tgt_box[6])
        tgt_box[0:3] = 0; tgt_box[6] = 0
        bike.append(tgt_pts)

    sedan = np.concatenate(sedan, axis=0)
    sedan = downsample_pts_sedan(sedan)

    suv = np.concatenate(suv, axis=0)
    suv = downsample_pts_suv(suv)
    
    pickup = np.concatenate(pickup, axis=0)
    pickup = downsample_pts_van(pickup)

    ped = np.concatenate(ped, axis=0)

    bike = np.concatenate(bike, axis=0)
    bike = downsample_pts_ped(bike)

    pickle.dump(sedan, open('sedan.pkl', 'wb'))
    pickle.dump(suv, open('suv.pkl', 'wb'))
    pickle.dump(pickup, open('pickup.pkl', 'wb'))
    pickle.dump(ped, open('ped.pkl', 'wb'))
    pickle.dump(bike, open('bike.pkl', 'wb'))

    # draw_scenes(sedan, draw_origin=True)
    # draw_scenes(suv, draw_origin=True)
    # draw_scenes(pickup, draw_origin=True)
        # if tgt_box[3] > 7:
        #     print(tgt_box)
        #     draw_scenes(tgt_pts, gt_boxes=np.array([tgt_box]), draw_origin=True)





if __name__ == "__main__":
    # cam_z_angle()
    # downsample_pts(np.load('D:/CODE/simgen/pts.npy'))
    # read_pcds(lidar_file)
    # li2global_color(lidar_file,db_records,100)
    # target_merge(instance_root, box_root)
    tokenlis = []
    count = 0
    for scenario in db_records.scenario_tag:
        if count % 50 == 0:
            token = scenario.lidar_pc_token
            tokenlis.append(token)
            count += 1
        else:
            count += 1
        

    with open("scenarios.yaml", "w") as f:
        yaml.dump(tokenlis, f, sort_keys=False)

    
    # print(db_records.lidar_pc[0].token)
