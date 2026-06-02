import os, traceback
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import cv2
import torch
import open3d
from copy import deepcopy

from nuplan.database.nuplan_db_orm.frame import Frame
from nuplan.database.utils.label.utils import raw_mapping
from nuplan.database.nuplan_db_orm.nuplandb_wrapper import NuPlanDBWrapper
from nuplan.database.nuplan_db.nuplan_scenario_queries import get_images_from_lidar_tokens
from mmcv.ops import points_in_boxes_part


# =========================
# =========================
NUPLAN_DATA_ROOT = r"./data/nuplan"
NUPLAN_MAPS_ROOT = r"./data/nuplan/nuplan-v1.1/maps"
NUPLAN_DB_FILES_ROOT = r"./data/nuplan/nuplan-v1.1/splits/mini"
NUPLAN_MAP_VERSION = "nuplan-maps-v1.0"
BLOB_PATH = r"./data/nuplan/nuplan-v1.1/sensor_blobs"

OUTPUT_ROOT = r"/cache/sWX1481336/bg_out"

DB_LIST = [
    "2021.05.12.22.28.35_veh-35_00620_01164",
]

CAMERALIST = ['CAM_F0','CAM_L0','CAM_L1','CAM_L2','CAM_B0','CAM_R2','CAM_R1']
GT_LIST = ['car','ped','bike']

START_IDX = 0
END_IDX = -1
STEP = 8

DOWNSAMPLE_VOXEL = 0.01

WORKERS = 24
# =========================


def li2global_color(lidar_root, log_db_name, db_record, idx, pbar):
    scene_token = db_record.lidar_pc[idx].token

    lid_record = db_record.lidar[0]
    l2e_t = lid_record.translation_np
    l2e_r = lid_record.quaternion.rotation_matrix

    ego_pose_record = db_record.lidar_pc[idx].ego_pose
    e2g_t = ego_pose_record.translation_np
    e2g_r = ego_pose_record.quaternion.rotation_matrix

    lidar_file = f"{lidar_root}/{scene_token}.pcd"
    boxes = db_record.lidar_pc[idx].boxes(frame=Frame.SENSOR)

    locs = np.array([b.center for b in boxes]).reshape(-1, 3)
    dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
    dims[:, [0, 1]] = dims[:, [1, 0]]
    rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(-1, 1)
    labels = np.array([b.label for b in boxes])
    names = np.array([raw_mapping["id2local"][l] for l in labels])
    gt_boxes = np.concatenate([locs, dims, rots], axis=1)

    pcd = open3d.io.read_point_cloud(lidar_file)
    points = np.asarray(pcd.points, dtype=np.float32)

    mask = labels != 0
    gt_boxes = gt_boxes[mask]
    names = names[mask]

    mask_gt = np.isin(names, GT_LIST)
    gt_boxes = gt_boxes[mask_gt]
    names = names[mask_gt]

    colored_pts = []

    for cam in CAMERALIST:
        db_path = os.path.join(NUPLAN_DB_FILES_ROOT, log_db_name + ".db")
        img_info = list(get_images_from_lidar_tokens(db_path, [scene_token], [cam]))
        cam_db = db_record.camera.select_one(channel=cam)
        if len(img_info) == 0:
            continue

        img_path = os.path.join(BLOB_PATH, img_info[0].filename_jpg)
        img = cv2.imread(img_path)
        if img is None:
            continue
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

        points_2d = (I @ points_camera.T).T
        points_2d[:, 0] /= points_2d[:, 2]
        points_2d[:, 1] /= points_2d[:, 2]
        points_2d = points_2d[:, :2]

        if points_2d.shape[0] == 0:
            continue
        points_2d = cv2.undistortPoints(points_2d.reshape(-1, 1, 2), I, distortion, None, I).reshape(-1, 2)

        H, W, _ = img.shape
        colors, valid_points = [], []
        for i, (u, v) in enumerate(points_2d):
            u, v = int(round(u)), int(round(v))
            if 0 <= u < W and 0 <= v < H:
                colors.append(img[v, u])
                valid_points.append(original_points[i])

        if len(valid_points) == 0:
            continue

        valid_points = np.asarray(valid_points, dtype=np.float32)
        colors = np.asarray(colors, dtype=np.float32)
        colored_pts.append(np.hstack((valid_points, colors)))

    if len(colored_pts) == 0:
        return None

    colored_pts = np.concatenate(colored_pts, axis=0)

    cut_box = deepcopy(gt_boxes)
    cut_box[:, 2] = cut_box[:, 2] - cut_box[:, 5] / 2
    cut_box[:, 4] = cut_box[:, 4] + 1
    cut_box[:, 3] = cut_box[:, 3] + 1

    m = cut_box[:, 5] < 3
    cut_box = cut_box[m]

    if cut_box.shape[0] > 0:
        cut_box_t = torch.from_numpy(cut_box).cuda().float()
        colored_t = torch.from_numpy(colored_pts).cuda().float()
        pts_in_boxes = points_in_boxes_part(colored_t[:, :3].unsqueeze(0), cut_box_t.unsqueeze(0)).squeeze(0).cpu().numpy()
        colored_t = colored_t[pts_in_boxes.squeeze() == -1]
        colored_pts = colored_t.cpu().numpy()

    colored_pts[:, :3] = colored_pts[:, :3] @ (l2e_r @ e2g_r).T + (l2e_t @ e2g_r + e2g_t)

    # Update progress bar for the current frame
    pbar.update(1)

    return colored_pts


def process_one_db(log_db_name: str):
    wrapper = NuPlanDBWrapper(
        data_root=NUPLAN_DATA_ROOT,
        map_root=NUPLAN_MAPS_ROOT,
        db_files=NUPLAN_DB_FILES_ROOT,
        map_version=NUPLAN_MAP_VERSION,
    )
    out_path = os.path.join(
        OUTPUT_ROOT, f"bg{log_db_name}_vox{DOWNSAMPLE_VOXEL}.npy"
    )
    if os.path.exists(out_path):
        return (log_db_name, True, f"skip (cached): {out_path}")
    # ==========================================
    db_record = wrapper.get_log_db(log_db_name)

    lidar_root = os.path.join(
        BLOB_PATH, log_db_name, "MergedPointCloud"
    )

    n = len(db_record.lidar_pc)
    s = max(0, START_IDX)
    e = n if END_IDX < 0 else min(n, END_IDX)

    pts_list = []
    # Create progress bar for each database processing
    with tqdm(total=(e - s) // STEP, desc=f"Processing {log_db_name}", ncols=100) as pbar:
        for idx in range(s, e, STEP):
            pts = li2global_color(lidar_root, log_db_name, db_record, idx, pbar)
            if pts is not None and len(pts) > 0:
                pts_list.append(pts)

    if len(pts_list) == 0:
        return (log_db_name, False, "no_points")

    pts = np.concatenate(pts_list, axis=0).astype(np.float32)

    # return (log_db_name, True, f"saved={pts.shape}")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    if not DOWNSAMPLE_VOXEL or DOWNSAMPLE_VOXEL <= 0:
        return (log_db_name, False, "DOWNSAMPLE_VOXEL<=0, nothing_saved")

    ply = open3d.geometry.PointCloud()
    ply.points = open3d.utility.Vector3dVector(pts[:, :3])
    ply.colors = open3d.utility.Vector3dVector(pts[:, 3:6] / 255.0)
    ply = ply.voxel_down_sample(voxel_size=float(DOWNSAMPLE_VOXEL))

    p = np.asarray(ply.points, dtype=np.float32)
    c = (np.asarray(ply.colors, dtype=np.float32) * 255.0)

    out_path = os.path.join(
        OUTPUT_ROOT, f"bg{log_db_name}_vox{DOWNSAMPLE_VOXEL}.npy"
    )
    np.save(out_path, np.hstack([p, c]).astype(np.float32))

    return (log_db_name, True, f"saved_vox={p.shape} -> {out_path}")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    futures = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        # Create a progress bar for the entire database processing
        pbar = tqdm(total=len(DB_LIST), desc="Processing DBs", ncols=100)
        for log_db_name in DB_LIST:
            futures.append(ex.submit(process_one_db, log_db_name))

        for fu in tqdm(as_completed(futures), total=len(futures), desc="Per-DB background"):
            try:
                name, ok, msg = fu.result()
                print(f"[{name}] ok={ok} {msg}")
                pbar.update(1)
            except Exception:
                print("[ERROR]\n", traceback.format_exc())
        pbar.close()
