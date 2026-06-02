import os
import pickle as pkl
from os import path as osp

import numba
import numpy as np
import multiprocessing as mp
import time
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.spatial.distance import cdist

from nuplan.database.nuplan_db_orm.nuplandb_wrapper import NuPlanDBWrapper
from nuplan.database.nuplan_db.nuplan_scenario_queries import (
    get_images_from_lidar_tokens,
    get_ego_state_for_lidarpc_token_from_db,
)
from nuplan.database.nuplan_db_orm.lidar_box import LidarBox
from nuplan.database.nuplan_db_orm.camera import Camera
from nuplan.database.nuplan_db_orm.frame import Frame
from nuplan.database.nuplan_db_orm.nuplandb import NuPlanDB
from nuplan.database.utils.label.utils import raw_mapping


from dwm.datasets.nuplan_splits import mini_train, mini_val 

##############################



##############################


cam_types = ['CAM_F0','CAM_L0','CAM_L1','CAM_L2','CAM_B0','CAM_R2','CAM_R1','CAM_R0']


@numba.njit
def rotate_round_z_axis(points: np.ndarray, angle: float):
    rotate_mat = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    return points @ rotate_mat


def _filter_blob_with_sensor(sensor_blobs_root, db_root, logs):
    """
    """
    sensor_exist = set()
    for shard in os.listdir(sensor_blobs_root):
        shard_path = osp.join(sensor_blobs_root, shard)
        if not (osp.isdir(shard_path) and shard.startswith("nuplan-v1.1")):
            continue
        for d in os.listdir(shard_path):
            if osp.isdir(osp.join(shard_path, d)):
                sensor_exist.add(d)

    db_exist = set(os.listdir(db_root))

    return [name for name in logs if name in sensor_exist and (name + ".db") in db_exist]


def obtain_sensor2top(cam_db, lid_record, ego_pose_record):
    sweep = {
        "sensor2ego_translation": cam_db.translation_np,
        "sensor2ego_rotation": cam_db.quaternion,
        "camera_intrinsics": cam_db.intrinsic,
        "distortion": cam_db.distortion
    }
    l2e_r_s = sweep["sensor2ego_rotation"]
    l2e_t_s = sweep["sensor2ego_translation"]

    l2e_t = lid_record.translation_np
    l2e_r_mat = lid_record.quaternion.rotation_matrix

    e2g_t = ego_pose_record.translation_np
    e2g_r = ego_pose_record.quaternion

    l2e_r_s_mat = l2e_r_s.rotation_matrix
    e2g_r_mat = e2g_r.rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T = (l2e_t_s @ e2g_r_mat.T + e2g_t) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T -= (
        e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        + l2e_t @ np.linalg.inv(l2e_r_mat).T
    )
    sweep["sensor2lidar_rotation"] = R.T
    sweep["sensor2lidar_translation"] = T
    return sweep


def get_ego_center_cache(db_record: NuPlanDB, data_root, db_name, step):
    sensor_record_len = len(db_record.lidar_pc)
    ego_center, lidar_pc_tokens, velos = [], [], []
    true_idx_to_ego_center_index = {}

    for idx in range(0, sensor_record_len, step):
        lidar_pc = db_record.lidar_pc[idx]

        image_infos = []
        for cam in cam_types:
            image_infos.extend(list(get_images_from_lidar_tokens(
                osp.join(data_root, db_name + ".db"), [lidar_pc.token], [cam]
            )))
        if len(image_infos) != len(cam_types):
            continue

        ego_state = get_ego_state_for_lidarpc_token_from_db(db_record.load_path, lidar_pc.token)
        true_idx_to_ego_center_index[idx] = len(ego_center)
        ego_center.append(ego_state.center.array)
        lidar_pc_tokens.append(lidar_pc.token)

        velocity = ego_state.dynamic_car_state.rear_axle_velocity_2d.array
        velo = rotate_round_z_axis(velocity, -ego_state.rear_axle.heading)
        n = np.linalg.norm(velo)
        velos.append(velo / n if n > 1e-6 else velo)

    return np.stack(ego_center), lidar_pc_tokens, np.stack(velos), true_idx_to_ego_center_index


def get_corresponding_infos(db_wrapper, data_root, db_record, db_name, val_set, step, progress_q=None):

    misalign = 0
    kept = 0

    sensor_record_len = len(db_record.lidar_pc)
    lid_record = db_record.lidar[0]
    log_record = db_record.log

    ego_center_cache, token_cache, velo_cache, idx_map = get_ego_center_cache(db_record, data_root, db_name, step)

    distance_matrix = cdist(ego_center_cache, ego_center_cache)
    ranges = [(2, 5), (5, 10), (10, 15)]

    train_info, val_info = [], []
    
    total_steps = (sensor_record_len + 1) // step  
    done_steps = 0
    last_report = time.time()
    
    for idx in range(0, sensor_record_len, step):
        
        done_steps += 1
        if progress_q is not None and (done_steps % 50 == 0):
            progress_q.put(("progress", db_name, done_steps, total_steps))
            last_report = time.time()
        if idx not in idx_map:
            continue

        lidar_pc = db_record.lidar_pc[idx]
        ego_pose_record = lidar_pc.ego_pose

        image_infos = []
        cam_dict = {}
        for cam in cam_types:
            cam_db: Camera = db_record.camera.select_one(channel=cam)
            cam_dict[cam] = obtain_sensor2top(cam_db, lid_record, ego_pose_record)
            image_infos.extend(list(get_images_from_lidar_tokens(
                osp.join(data_root, db_name + ".db"), [lidar_pc.token], [cam]
            )))

        if len(image_infos) != len(cam_types):
            misalign += 1
            continue

        kept += 1

        info = {
            "lidar2ego": lid_record.trans_matrix,
            "db_name": db_name,
            "cam": cam_dict,
            "timestamp": ego_pose_record.timestamp,
            "location": log_record.location,
            "date": log_record.date,
            "lidarpc_token": lidar_pc.token,
            "scene_token": lidar_pc.scene_token,
            "ego2global": ego_pose_record.trans_matrix,
        }

        lidar_boxes = db_record.session.query(LidarBox).filter(
            LidarBox.lidar_pc_token == lidar_pc.token
        ).all()
        track_token = np.array([box.track_token for box in lidar_boxes])

        qw, qx, qy, qz = ego_pose_record.qw, ego_pose_record.qx, ego_pose_record.qy, ego_pose_record.qz
        info["egopose"] = [
            ego_pose_record.x, ego_pose_record.y, ego_pose_record.z,
            np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))
        ]

        info["img_filename"] = [i.filename_jpg for i in image_infos]
        info["img_tokens"] = [i.token for i in image_infos]

        info["ego_feats"] = np.array([
            ego_pose_record.vx, ego_pose_record.vy,
            ego_pose_record.acceleration_x, ego_pose_record.acceleration_y,
            ego_pose_record.angular_rate_z
        ])

        # ref tokens
        info["ref_front_tokens"], info["ref_rear_tokens"] = [], []
        dis_index = idx_map[idx]
        for lower, upper in ranges:
            indices = np.where(
                (distance_matrix[dis_index] >= lower) & (distance_matrix[dis_index] < upper)
            )[0]
            indices = indices[indices != dis_index]

            pos, neg = [], []
            for ref_idx in indices:
                dv = ego_center_cache[ref_idx] - ego_center_cache[dis_index]
                dot = np.dot(dv, velo_cache[dis_index])
                if dot > 0:
                    pos.append(ref_idx)
                elif dot < 0:
                    neg.append(ref_idx)

            info["ref_front_tokens"].append([token_cache[i] for i in pos] if pos else None)
            info["ref_rear_tokens"].append([token_cache[i] for i in neg] if neg else None)

        # boxes
        boxes = lidar_pc.boxes(frame=Frame.SENSOR)
        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        dims_line = dims.copy()
        dims[:, [0, 1]] = dims[:, [1, 0]]
        rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(-1, 1)

        labels = np.array([b.label for b in boxes])
        names = np.array([raw_mapping["id2local"][l] for l in labels])

        gt_boxes = np.concatenate([locs, dims, rots], axis=1)
        gt_line = np.concatenate([locs, dims_line, -rots - np.pi / 2], axis=1)  # z?
        gt_line[:,2] -= 1/2 * gt_line[:,5]

        mask = labels != 0
        info["gt_boxes"] = gt_boxes[mask]
        info["gt_line"] = gt_line[mask]
        info["gt_names"] = names[mask]
        info["track_token"] = track_token[mask]

        (val_info if db_name in val_set else train_info).append(info)
    
    if progress_q is not None:
        progress_q.put(("done", db_name, done_steps, total_steps))

    return train_info, val_info, {"db": db_name, "kept": kept, "misalign": misalign}



def fill_train_val_info(data_root, map_root, map_version, logs, val_set, step=2, workers=12):

    db_paths = [osp.join(data_root, x + ".db") for x in logs]
    db_wrapper = NuPlanDBWrapper(data_root, map_root, db_paths, map_version)
    db_records = [db_wrapper.get_log_db(db_name) for db_name in logs]

    manager = mp.Manager()
    progress_q = manager.Queue()
    
    total_steps_all = sum((len(r.lidar_pc) + 1) // step for r in db_records)

    p_submit = tqdm(total=len(logs), desc="submit", unit="log")
    p_steps = tqdm(total=total_steps_all, desc="overall frames", unit="step")
    p_done = tqdm(total=len(logs), desc="done logs", unit="log")

    last_done = {name: 0 for name in logs}  
    train_infos, val_infos = [], []
    futures = []

    with ProcessPoolExecutor(max_workers=workers) as ex:
        for db_record, db_name in zip(db_records, logs):
            futures.append(ex.submit(
                get_corresponding_infos,
                db_wrapper, data_root, db_record, db_name, val_set, step, progress_q
            ))
            p_submit.update(1)
        p_submit.close()

        remaining = set(futures)
        while remaining:
            while not progress_q.empty():
                typ, name, done_steps, total_steps = progress_q.get()
                prev = last_done.get(name, 0)
                if done_steps > prev:
                    p_steps.update(done_steps - prev)
                    last_done[name] = done_steps
                p_steps.set_postfix_str(f"{name} {done_steps}/{total_steps}")
                if typ == "done":
                    p_done.update(1)

            finished = {fu for fu in list(remaining) if fu.done()}
            for fu in finished:
                tr, va, stat = fu.result()
                train_infos.extend(tr)
                val_infos.extend(va)
                remaining.remove(fu)

            time.sleep(0.05)

    p_steps.close()
    p_done.close()
    return train_infos, val_infos


def create_nuplan_infos(version, sensor_blobs_root, data_root, map_root, train_logs, val_logs, out_path,
                        map_version="nuplan-maps-v1.0", step=2, workers=12):
    train_logs = _filter_blob_with_sensor(sensor_blobs_root, data_root, train_logs)
    val_logs   = _filter_blob_with_sensor(sensor_blobs_root, data_root, val_logs)

    
    logs = train_logs + val_logs
    val_set = set(val_logs)

    train_infos, val_infos = fill_train_val_info(
        data_root, map_root, map_version, logs, val_set, step=step, workers=workers
    )

    meta = dict(version=version)
    train = dict(infos=train_infos, metadata=meta)
    val = dict(infos=val_infos, metadata=meta)

    os.makedirs(out_path, exist_ok=True)
    p_train = osp.join(out_path, f"{version}_infos_train.pkl")
    p_val = osp.join(out_path, f"{version}_infos_val.pkl")

    with open(p_train, "wb") as f:
        pkl.dump(train, f)
    with open(p_val, "wb") as f:
        pkl.dump(val, f)

    print("saved:", p_train, p_val)
    # print("train samples:", len(train_infos), "val samples:", len(val_infos))


if __name__ == "__main__":
    sensor_blobs_root = "./data/nuplan/nuplan-v1.1/sensor_blobs"

    dataset_root = "./data/nuplan/nuplan-v1.1"
    data_root = osp.join(dataset_root, "plan_data", "mini")
    map_root = osp.join(dataset_root, "maps")
    out_path = "./data/nuplan"

    create_nuplan_infos(
        "mini",
        sensor_blobs_root=sensor_blobs_root,
        data_root=data_root,
        map_root=map_root,
        train_logs=mini_train,
        val_logs=mini_val,
        out_path=out_path,
        step=2,
        workers=36
    )
