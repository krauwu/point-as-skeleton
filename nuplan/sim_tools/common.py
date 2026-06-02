import numba
from pyquaternion import Quaternion
from nuplan.database.nuplan_db.query_session import execute_many, execute_one
from pathlib import Path
import torch
from typing import Deque

import cv2
import numpy as np
import numpy.typing as npt
import torch
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.planning.simulation.planner.ml_planner.transform_utils import (
    _get_fixed_timesteps,
    _get_velocity_and_acceleration,
    _se2_vel_acc_to_ego_state,
)

@numba.njit
def rotate_round_z_axis(points: np.ndarray, angle: float):
    c = np.float32(np.cos(angle))
    s = np.float32(np.sin(angle))

    rotate_mat = np.array(
        [[c, -s],
         [s,  c]],
        dtype=np.float32
    )

    return points @ rotate_mat

def resize_img(img, resize_lim, final_shape):
    W, H = img.size
    fH, fW = final_shape
    resize = resize_lim
    resize_dims = (int(W * resize), int(H * resize))
    newW, newH = resize_dims
    crop_h = newH - fH
    crop_w = newW - fW
    crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)

    img = img.resize(resize_dims)
    img = img.crop(crop)
    return img

def get_aug_mat(img, resize_lim, final_shape):
    W, H = img.size
    fH, fW = final_shape
    resize = resize_lim
    resize_dims = (int(W * resize), int(H * resize))
    newW, newH = resize_dims
    crop_h = newH - fH
    crop_w = newW - fW
    crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)

    aug_mat = torch.eye(4)
    aug_mat[:2, :2] *= resize
    aug_mat[:2, 3] -= torch.Tensor(crop[:2])
    return aug_mat
    

def _pil_or_np_to_rgb_u8(img) -> np.ndarray:
    """PIL or np -> HWC RGB uint8"""
    if img is None:
        return None
    if hasattr(img, "convert"):  # PIL
        return np.array(img.convert("RGB"), dtype=np.uint8)

    arr = np.array(img, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return arr


def _make_cam_row_bgr(cam_list, H: int, W: int, V: int) -> np.ndarray:
    """cam_list(len>=V) -> (H, V*W, 3) BGR uint8"""
    tiles = []
    for v in range(V):
        if cam_list is None or v >= len(cam_list) or cam_list[v] is None:
            tile = np.zeros((H, W, 3), dtype=np.uint8)
        else:
            rgb = _pil_or_np_to_rgb_u8(cam_list[v])
            rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
            tile = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        tiles.append(tile)
    return np.hstack(tiles)


def _tensor_chw_to_bgr_u8(x: torch.Tensor) -> np.ndarray:
    # x: [3,H,W], float/half/uint8; range in [0,1] or [-1,1]
    if x.dtype != torch.uint8:
        x = x.float()
        if x.min() < 0:
            x = (x + 1.0) * 0.5
        x = x.clamp(0.0, 1.0)
        x = (x * 255.0 + 0.5).to(torch.uint8)

    # CHW RGB -> HWC BGR
    x = x.permute(1, 2, 0).contiguous()  # HWC RGB
    arr = x.cpu().numpy()
    return arr[:, :, ::-1]  # BGR

def _make_triplet_v_frame_bgr(
    cam8_list,
    proj_sem_tvchw: torch.Tensor,  # [T,V,3,H,W]
    proj_clr_tvchw: torch.Tensor,  # [T,V,3,H,W]
    t_idx: int,
    raw_cam8_list=None,
) -> np.ndarray:
    H = int(proj_sem_tvchw.shape[-2])
    W = int(proj_sem_tvchw.shape[-1])
    V = int(proj_sem_tvchw.shape[1])

    sem_v = proj_sem_tvchw[t_idx]  # [V,3,H,W]
    clr_v = proj_clr_tvchw[t_idx]  # [V,3,H,W]

    tiles = []
    for v in range(V):
        if cam8_list is None or v >= len(cam8_list) or cam8_list[v] is None:
            base_bgr = np.zeros((H, W, 3), dtype=np.uint8)
        else:
            base_rgb = np.array(cam8_list[v].convert("RGB"), dtype=np.uint8)
            base_bgr = cv2.cvtColor(cv2.resize(base_rgb, (W, H)), cv2.COLOR_RGB2BGR)

        # --- row2/3: proj tensors ---
        sem_bgr = _tensor_chw_to_bgr_u8(sem_v[v])
        clr_bgr = _tensor_chw_to_bgr_u8(clr_v[v])

        if raw_cam8_list is None or v >= len(raw_cam8_list) or raw_cam8_list[v] is None:
            raw_bgr = np.zeros((H, W, 3), dtype=np.uint8)
        else:
            raw_rgb = np.array(raw_cam8_list[v].convert("RGB"), dtype=np.uint8)
            raw_bgr = cv2.cvtColor(cv2.resize(raw_rgb, (W, H)), cv2.COLOR_RGB2BGR)

        quad = np.vstack([base_bgr, sem_bgr, clr_bgr, raw_bgr])
        tiles.append(quad)

    return np.hstack(tiles)


# def _make_triplet_v_frame_bgr(
#     cam8_list,                 # List[PIL] or None, len=V
#     proj_sem_tvchw: torch.Tensor,  # [T,V,3,224,400]
#     proj_clr_tvchw: torch.Tensor,  # [T,V,3,224,400]
#     t_idx: int,
# ) -> np.ndarray:
#     H = int(proj_sem_tvchw.shape[-2])
#     W = int(proj_sem_tvchw.shape[-1])
#     V = int(proj_sem_tvchw.shape[1])

#     sem_v = proj_sem_tvchw[t_idx]  # [V,3,H,W]
#     clr_v = proj_clr_tvchw[t_idx]  # [V,3,H,W]

#     tiles = []
#     for v in range(V):
#         if cam8_list is None or v >= len(cam8_list) or cam8_list[v] is None:
#             base_bgr = np.zeros((H, W, 3), dtype=np.uint8)
#         else:
#             base_rgb = np.array(cam8_list[v].convert("RGB"), dtype=np.uint8)
#             base_bgr = cv2.cvtColor(cv2.resize(base_rgb, (W, H)), cv2.COLOR_RGB2BGR)

#         sem_bgr = _tensor_chw_to_bgr_u8(sem_v[v])
#         clr_bgr = _tensor_chw_to_bgr_u8(clr_v[v])

#         tiles.append(triplet)




def get_transmat_for_lidarpc_token_from_db(log_file: str, token: str):
    query = """
        SELECT  ep.x,
                ep.y,
                ep.z,
                ep.qw,
                ep.qx,
                ep.qy,
                ep.qz,
                -- ego_pose and lidar_pc timestamps are not the same, even when linked by token!
                -- use lidar_pc timestamp for backwards compatibility.
                lp.timestamp,
                ep.vx,
                ep.vy,
                ep.acceleration_x,
                ep.acceleration_y
        FROM ego_pose AS ep
        INNER JOIN lidar_pc AS lp
            ON lp.ego_pose_token = ep.token
        WHERE lp.token = ?
    """

    row = execute_one(query, (bytearray.fromhex(token),), log_file)
    if row is None:
        return None

    q = Quaternion(row["qw"], row["qx"], row["qy"], row["qz"])
    translation = np.array([row["x"], row["y"], row["z"]])
    trans_mat = q.transformation_matrix
    trans_mat[:3, 3] = translation
    return trans_mat

def obtain_sensor2top(cam_db, lid_record, ego2global):
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

    e2g_t = ego2global[:3, 3].reshape(3)
    e2g_r = ego2global[:3, :3]

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego->lidar
    l2e_r_s_mat = l2e_r_s.rotation_matrix
    e2g_r_mat = e2g_r
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
    sweep["sensor2lidar_rotation"] = R.T  # points @ R.T + T
    sweep["sensor2lidar_translation"] = T
    
    return sweep

def global_trajectory_to_states(
    global_trajectory: npt.NDArray[np.float32],
    ego_history: Deque[EgoState],
    future_horizon: float,
    step_interval: float,
    include_ego_state: bool = True,
):
    ego_state = ego_history[-1]
    timesteps = _get_fixed_timesteps(ego_state, future_horizon, step_interval)
    global_states = [StateSE2.deserialize(pose) for pose in global_trajectory]

    velocities, accelerations = _get_velocity_and_acceleration(
        global_states, ego_history, timesteps
    )
    agent_states = [
        _se2_vel_acc_to_ego_state(
            state,
            velocity,
            acceleration,
            timestep,
            ego_state.car_footprint.vehicle_parameters,
        )
        for state, velocity, acceleration, timestep in zip(
            global_states, velocities, accelerations, timesteps
        )
    ]

    if include_ego_state:
        agent_states.insert(0, ego_state)

    return agent_states

