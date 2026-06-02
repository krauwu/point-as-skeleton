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
DST_DIR = "./data/nuplan/actors/track_actor_down"

WORKERS = 16
SKIP_EXISTING = True

SEED = 12345
RATIO_MIN = 0.05
RATIO_MAX = 0.80
RATIO_LOG_UNIFORM = True

MIN_KEEP_POINTS = 200
MIN_VOXEL = 1e-2
MAX_VOXEL = 1.0

ENABLE_FINAL_RANDOM_TRIM = False
# =========================


def make_rng_from_seed_and_name(seed, name):
    s = int(seed)
    if s < 0:
        return np.random.default_rng()
    h = (hash(name) & 0xFFFFFFFF)
    return np.random.default_rng(s ^ h)


def choose_random_ratio_geometric(rng, rmin, rmax, log_uniform=True):
    a = float(rmin)
    b = float(rmax)
    a = max(1e-6, min(a, b))
    b = max(a, b)
    if log_uniform:
        la = math.log(a)
        lb = math.log(b)
        return float(math.exp(rng.uniform(la, lb)))
    return float(rng.uniform(a, b))


def estimate_voxel_from_target_count(xyz, target_count, min_voxel, max_voxel):
    """
    voxel ~ (volume / target_count)^(1/3)
    """
    if xyz.shape[0] == 0:
        return float(min_voxel)

    target = int(target_count)
    target = max(1, target)

    xyz_min = xyz.min(axis=0)
    xyz_max = xyz.max(axis=0)
    extent = (xyz_max - xyz_min).astype(np.float64)

    ex = float(extent[0])
    ey = float(extent[1])
    ez = float(extent[2])

    vol = max(ex * ey * ez, 1e-12)
    voxel = (vol / float(target)) ** (1.0 / 3.0)

    voxel = max(float(min_voxel), min(float(max_voxel), float(voxel)))
    return float(voxel)


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


def voxel_downsample_xyzrgb_to_target(pts6, target_count, min_voxel, max_voxel):
    xyz = pts6[:, :3].astype(np.float64)
    voxel = estimate_voxel_from_target_count(xyz, target_count, min_voxel, max_voxel)
    pcd = to_o3d_pcd_xyzrgb(pts6)
    pcd2 = pcd.voxel_down_sample(voxel_size=float(voxel))
    out = from_o3d_pcd_xyzrgb(pcd2)
    return out, float(voxel)


def random_trim_no_replace(pts6, rng, keep_count):
    n = int(pts6.shape[0])
    k = int(keep_count)
    if k >= n:
        return pts6
    idx = rng.choice(n, size=k, replace=False)
    return pts6[idx].astype(np.float32)


def process_one_file(in_path, out_path):
    try:
        pts = np.load(in_path)
        if pts.ndim != 2 or pts.shape[1] != 6:
            return (in_path, False, 0, 0, 0.0, 0.0, "bad shape")

        n_in = int(pts.shape[0])
        if n_in == 0:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            np.save(out_path, pts.astype(np.float32))
            return (in_path, True, 0, 0, 1.0, 0.0, "empty")

        rng = make_rng_from_seed_and_name(SEED, os.path.basename(in_path))

        ratio = choose_random_ratio_geometric(rng, RATIO_MIN, RATIO_MAX, log_uniform=RATIO_LOG_UNIFORM)
        target = int(round(float(n_in) * float(ratio)))
        target = max(int(MIN_KEEP_POINTS), target)
        target = min(n_in, target)

        out, voxel = voxel_downsample_xyzrgb_to_target(
            pts.astype(np.float32),
            target_count=target,
            min_voxel=MIN_VOXEL,
            max_voxel=MAX_VOXEL,
        )

        if ENABLE_FINAL_RANDOM_TRIM and out.shape[0] > target:
            out = random_trim_no_replace(out, rng, target)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, out.astype(np.float32))
        return (in_path, True, n_in, int(out.shape[0]), float(ratio), float(voxel), "ok")

    except Exception as e:
        return (in_path, False, 0, 0, 0.0, 0.0, repr(e))


def main():
    os.makedirs(DST_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(SRC_DIR, "*.npy")))
    if len(files) == 0:
        print(f"[WARN] no npy found in {SRC_DIR}")
        return

    tasks = []
    with ProcessPoolExecutor(max_workers=int(WORKERS)) as ex:
        for in_path in files:
            name = os.path.basename(in_path)
            out_path = os.path.join(DST_DIR, name)
            if SKIP_EXISTING and os.path.isfile(out_path):
                continue
            tasks.append(ex.submit(process_one_file, in_path, out_path))

        ok_cnt = 0
        bad_cnt = 0
        in_sum = 0
        out_sum = 0

        ratio_min = 1.0
        ratio_max = 0.0
        vox_min = float("inf")
        vox_max = 0.0

        for fu in tqdm(as_completed(tasks), total=len(tasks), ncols=120):
            in_path, ok, n_in, n_out, ratio, voxel, msg = fu.result()
            if ok:
                ok_cnt += 1
                in_sum += int(n_in)
                out_sum += int(n_out)
                ratio_min = min(ratio_min, float(ratio))
                ratio_max = max(ratio_max, float(ratio))
                if voxel > 0.0:
                    vox_min = min(vox_min, float(voxel))
                    vox_max = max(vox_max, float(voxel))
            else:
                bad_cnt += 1
                print(f"[FAIL] {in_path} -> {msg}")

    keep_ratio = float(out_sum) / float(max(1, in_sum))
    print(f"[DONE] ok={ok_cnt}, fail={bad_cnt}")
    print(f"[STATS] total_in={in_sum}, total_out={out_sum}, keep_ratio={keep_ratio:.6f}")
    print(f"[RATIO] per-file ratio in [{ratio_min:.4f}, {ratio_max:.4f}]  (log_uniform={RATIO_LOG_UNIFORM})")
    if vox_min < float("inf"):
        print(f"[VOXEL] per-file voxel in [{vox_min:.6g}, {vox_max:.6g}]")
    print(f"[SAVE] {DST_DIR}")


if __name__ == "__main__":
    main()
