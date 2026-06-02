#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import math
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import open3d as o3d
from tqdm import tqdm


# =========================
# HARDCODE CONFIG
# =========================
SRC_DIR = "./data/nuplan/actors/track_actor_ori"
DST_DIR = "./data/nuplan/actors/track_actor_cut"

WORKERS = 16
SKIP_EXISTING = True

IMG_W = 1024
IMG_H = 768
FOV_X_DEG = 90.0
KEEP_K = 1

SEED = -1
SPREAD_DEG = 30.0
EPS = 0.02
RATIO_THRESH = 1.2

RADIUS_SCALE = 2.5
MIN_RADIUS = 6.0
HEIGHT_OFFSET = 0.1

DOWNSAMPLE_THRESHOLD = 300_000
DOWNSAMPLE_VOX = 0.02
# =========================


def make_rng_from_seed_and_name(seed, name):
    s = int(seed)
    if s < 0:
        return np.random.default_rng()
    h = (hash(name) & 0xFFFFFFFF)
    return np.random.default_rng(s ^ h)


def pick_theta_side_front_back(points_xyzrgb, rng, spread_deg=20.0, eps=0.02, ratio_thresh=1.2):
    xyz = points_xyzrgb[:, :3].astype(np.float32)
    y = xyz[:, 1]

    eps_f = float(eps)
    n_left = int(np.sum(y > eps_f))
    n_right = int(np.sum(y < -eps_f))

    centers_deg = np.array([45.0, -45.0, 135.0, -135.0], dtype=np.float32)

    if (n_left + n_right) > 0:
        big = max(n_left, n_right)
        small = max(1, min(n_left, n_right))
        if float(big) / float(small) >= float(ratio_thresh):
            if n_left > n_right:
                centers_deg = np.array([45.0, 135.0], dtype=np.float32)
            else:
                centers_deg = np.array([-45.0, -135.0], dtype=np.float32)

    center = float(rng.choice(centers_deg))
    jitter = float(rng.uniform(-float(spread_deg), float(spread_deg)))
    return float(np.deg2rad(center + jitter))


def choose_radius_from_points(points_xyzrgb, scale=2.5, min_radius=6.0):
    xyz = points_xyzrgb[:, :3].astype(np.float32)
    if xyz.shape[0] == 0:
        return float(min_radius)
    r_xy = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
    r = float(np.max(r_xy)) * float(scale)
    return max(r, float(min_radius))


def build_virtual_intrinsics(img_w, img_h, fov_x_deg):
    fov_x = float(fov_x_deg) * math.pi / 180.0
    fx = 0.5 * float(img_w) / math.tan(0.5 * fov_x)
    fy = fx
    cx = 0.5 * float(img_w)
    cy = 0.5 * float(img_h)
    K = np.array(
        [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return K


def safe_normalize(v):
    n = float(np.linalg.norm(v)) + 1e-12
    return (v / n).astype(np.float32)


def world_to_camera_matrix(cam_pos, target=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                           up=np.array([0.0, 0.0, 1.0], dtype=np.float32)):
    """
    """
    cam_pos = cam_pos.astype(np.float32)
    target = target.astype(np.float32)
    up = up.astype(np.float32)

    forward = safe_normalize(target - cam_pos)
    right = safe_normalize(np.cross(forward, up))
    true_up = np.cross(right, forward).astype(np.float32)

    # camera axes in world: [right, up, forward]
    R_wc = np.stack([right, true_up, forward], axis=1).astype(np.float32)  # world->camera inverse
    R_cw = R_wc.T.astype(np.float32)
    t_cw = (-R_cw @ cam_pos.reshape(3, 1)).reshape(3).astype(np.float32)
    return R_cw, t_cw


def zbuffer_project_visible_points(points_xyzrgb, cam_pos, K, img_w, img_h, keep_k=1):
    """
    """
    pts6 = points_xyzrgb.astype(np.float32)
    xyz = pts6[:, :3]

    Rcw, tcw = world_to_camera_matrix(cam_pos)
    pc = (xyz @ Rcw.T) + tcw.reshape(1, 3)  # (N,3)
    z = pc[:, 2]

    m0 = z > 1e-6
    if not np.any(m0):
        return np.empty((0, 6), dtype=np.float32)

    pc = pc[m0]
    pts6 = pts6[m0]
    z = z[m0]

    fx = float(K[0, 0]); fy = float(K[1, 1])
    cx = float(K[0, 2]); cy = float(K[1, 2])

    u = (fx * (pc[:, 0] / z) + cx)
    v = (fy * (pc[:, 1] / z) + cy)

    ui = np.floor(u).astype(np.int32)
    vi = np.floor(v).astype(np.int32)

    m1 = (ui >= 0) & (ui < int(img_w)) & (vi >= 0) & (vi < int(img_h))
    if not np.any(m1):
        return np.empty((0, 6), dtype=np.float32)

    ui = ui[m1]
    vi = vi[m1]
    z = z[m1]
    pts6 = pts6[m1]

    key = vi.astype(np.int64) * int(img_w) + ui.astype(np.int64)

    order = np.lexsort((z, key))
    key_sorted = key[order]

    change = np.ones_like(key_sorted, dtype=bool)
    change[1:] = key_sorted[1:] != key_sorted[:-1]
    group_start = np.where(change)[0]
    group_end = np.r_[group_start[1:], len(key_sorted)]

    kk = int(keep_k)
    picked_list = []
    for s, e in zip(group_start, group_end):
        cnt = int(e - s)
        if cnt <= 0:
            continue
        picked_list.append(order[s : s + min(kk, cnt)])

    if len(picked_list) == 0:
        return np.empty((0, 6), dtype=np.float32)

    picked = np.concatenate(picked_list, axis=0)
    return pts6[picked].astype(np.float32)


def to_o3d_pcd_xyzrgb(pts6):
    pcd = o3d.geometry.PointCloud()
    xyz = pts6[:, :3].astype(np.float64)
    rgb = pts6[:, 3:6].astype(np.float32)
    if rgb.size > 0 and rgb.max() > 1.0:
        rgb = (rgb / 255.0).clip(0.0, 1.0)
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
    return pcd


def from_o3d_pcd_xyzrgb(pcd):
    xyz = np.asarray(pcd.points, dtype=np.float32)
    col = np.asarray(pcd.colors, dtype=np.float32)
    col = (col * 255.0).clip(0.0, 255.0).astype(np.float32)
    if xyz.shape[0] == 0:
        return np.empty((0, 6), dtype=np.float32)
    return np.hstack([xyz, col]).astype(np.float32)


def voxel_downsample_xyzrgb(pts6, voxel):
    pcd = to_o3d_pcd_xyzrgb(pts6)
    pcd2 = pcd.voxel_down_sample(voxel_size=float(voxel))
    return from_o3d_pcd_xyzrgb(pcd2)


def process_one_file(in_path, out_path, K):
    try:
        pts = np.load(in_path)
        if pts.ndim != 2 or pts.shape[1] != 6:
            return (in_path, False, 0, 0, "bad shape")

        n_in = int(pts.shape[0])
        if n_in == 0:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            np.save(out_path, pts.astype(np.float32))
            return (in_path, True, 0, 0, "empty")

        rng = make_rng_from_seed_and_name(SEED, os.path.basename(in_path))

        theta = pick_theta_side_front_back(
            pts, rng=rng, spread_deg=SPREAD_DEG, eps=EPS, ratio_thresh=RATIO_THRESH
        )
        cam_r = choose_radius_from_points(pts, scale=RADIUS_SCALE, min_radius=MIN_RADIUS)

        z_max = float(np.max(pts[:, 2]))
        cam_z = z_max + float(HEIGHT_OFFSET)

        cam_pos = np.array(
            [cam_r * math.cos(theta), cam_r * math.sin(theta), cam_z],
            dtype=np.float32,
        )

        cut = zbuffer_project_visible_points(
            pts.astype(np.float32),
            cam_pos=cam_pos,
            K=K,
            img_w=IMG_W,
            img_h=IMG_H,
            keep_k=KEEP_K,
        )

        if cut.shape[0] > int(DOWNSAMPLE_THRESHOLD):
            cut = voxel_downsample_xyzrgb(cut, DOWNSAMPLE_VOX)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, cut.astype(np.float32))
        return (in_path, True, n_in, int(cut.shape[0]), "ok")
    except Exception as e:
        return (in_path, False, 0, 0, repr(e))


def main():
    os.makedirs(DST_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(SRC_DIR, "*.npy")))
    if len(files) == 0:
        print(f"[WARN] no npy found in {SRC_DIR}")
        return

    K = build_virtual_intrinsics(IMG_W, IMG_H, FOV_X_DEG)

    tasks = []
    with ProcessPoolExecutor(max_workers=int(WORKERS)) as ex:
        for in_path in files:
            name = os.path.basename(in_path)
            out_path = os.path.join(DST_DIR, name)

            if SKIP_EXISTING and os.path.isfile(out_path):
                continue

            tasks.append(ex.submit(process_one_file, in_path, out_path, K))

        ok_cnt = 0
        bad_cnt = 0
        in_sum = 0
        out_sum = 0

        for fu in tqdm(as_completed(tasks), total=len(tasks), ncols=120):
            in_path, ok, n_in, n_out, msg = fu.result()
            if ok:
                ok_cnt += 1
                in_sum += int(n_in)
                out_sum += int(n_out)
            else:
                bad_cnt += 1
                print(f"[FAIL] {in_path} -> {msg}")

    keep_ratio = float(out_sum) / float(max(1, in_sum))
    print(f"[DONE] ok={ok_cnt}, fail={bad_cnt}")
    print(f"[STATS] total_in={in_sum}, total_out={out_sum}, keep_ratio={keep_ratio:.6f}")
    print(f"[SAVE] {DST_DIR}")


if __name__ == "__main__":
    main()
