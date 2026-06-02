import os
import json
import pickle
import numpy as np
import cv2
import torch

from PIL import Image, ImageDraw
from pyquaternion import Quaternion

from dwm.datasets.common import (
    depth_to_logbins_u16, depth_to_linbins_u16,
    visualize_bins_u16,
    downsample_depth_blockwise, downsample_clr_blockwise,
)

cv2.setNumThreads(0)

def _expand_uv(u, v, z, r, H, W):
    if r <= 0:
        m = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        return u[m], v[m], z[m]

    d = np.arange(-r, r + 1, dtype=np.int32)
    dx, dy = np.meshgrid(d, d)
    dx = dx.reshape(-1); dy = dy.reshape(-1)

    uu = (u[:, None] + dx[None, :]).reshape(-1)
    vv = (v[:, None] + dy[None, :]).reshape(-1)
    zz = np.repeat(z, dx.size)

    m = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
    return uu[m], vv[m], zz[m]

# -------- core: project depth + sem(one-hot) + clr(rgb) --------

def project_depth_only(
    pts_xyz: np.ndarray,
    image_from_lidar: np.ndarray,
    ori_hw,
    *, invalid_depth=-300.0, splat=None
):
    H, W = int(ori_hw[0]), int(ori_hw[1])
    if splat is None:
        splat = [(1e9, 0)]

    if pts_xyz is None or pts_xyz.shape[0] == 0:
        return np.full((H, W), float(invalid_depth), np.float32)

    xyz1 = np.concatenate([pts_xyz.astype(np.float32), np.ones((pts_xyz.shape[0], 1), np.float32)], 1)
    p = xyz1 @ image_from_lidar.T
    z = p[:, 2]
    m = z > 1e-5
    p, z = p[m], z[m]

    u = (p[:, 0] / z).astype(np.int32)
    v = (p[:, 1] / z).astype(np.int32)

    depth = np.full((H, W), np.inf, np.float32)

    z2, u2, v2 = z.copy(), u.copy(), v.copy()
    for zmax, r in splat:
        mm = z2 <= zmax
        if not np.any(mm):
            continue
        uu, vv, zz = _expand_uv(u2[mm], v2[mm], z2[mm], r, H, W)
        np.minimum.at(depth.reshape(-1), vv * W + uu, zz)
        z2, u2, v2 = z2[~mm], u2[~mm], v2[~mm]
        if z2.size == 0:
            break

    depth[~np.isfinite(depth)] = float(invalid_depth)
    return depth


def project_sem_only(
    pts_xyz: np.ndarray,
    pts_sem: np.ndarray,
    image_from_lidar: np.ndarray,
    ori_hw,
    *, splat=None, n_actor_classes=3
):
    """
    returns sem: (H,W,K) uint8 {0,1}
    """
    H, W = int(ori_hw[0]), int(ori_hw[1])
    K = int(n_actor_classes)
    if splat is None:
        splat = [(1e9, 0)]

    sem = np.zeros((H, W, K), np.uint8)
    if pts_xyz is None or pts_xyz.shape[0] == 0 or pts_sem is None or pts_sem.shape[0] == 0:
        return sem

    xyz1 = np.concatenate([pts_xyz.astype(np.float32), np.ones((pts_xyz.shape[0], 1), np.float32)], 1)
    p = xyz1 @ image_from_lidar.T
    z = p[:, 2]
    m = z > 1e-5
    p, z = p[m], z[m]
    sid = pts_sem[m].astype(np.int32)

    # keep only 1..K
    good = (sid >= 1) & (sid <= K)
    if not np.any(good):
        return sem

    u = (p[:, 0] / z).astype(np.int32)[good]
    v = (p[:, 1] / z).astype(np.int32)[good]
    zc = z[good].astype(np.float32)
    sid = sid[good].astype(np.int32)

    flat, zz, sid2 = _collect_splat(u, v, zc, sid, H, W, splat)
    if flat.size == 0:
        return sem

    uf, _, sf = _zbuffer_first(flat, zz, sid2.astype(np.int32))
    sf = sf - 1
    ok = (sf >= 0) & (sf < K)
    sem.reshape(-1, K)[uf[ok], sf[ok]] = 1
    return sem


def project_clr_only(
    clr_xyz: np.ndarray,
    clr_rgb: np.ndarray,
    image_from_lidar: np.ndarray,
    ori_hw,
    *, splat=None
):
    H, W = int(ori_hw[0]), int(ori_hw[1])
    if splat is None:
        splat = [(1e9, 0)]

    clr = np.zeros((H, W, 3), np.uint8)
    if clr_xyz is None or clr_xyz.shape[0] == 0:
        return clr

    xyz1c = np.concatenate([clr_xyz.astype(np.float32), np.ones((clr_xyz.shape[0], 1), np.float32)], 1)
    pc = xyz1c @ image_from_lidar.T
    zc = pc[:, 2]
    mc = zc > 1e-5
    pc, zc, rgb = pc[mc], zc[mc], clr_rgb[mc].astype(np.uint8)

    uc = (pc[:, 0] / zc).astype(np.int32)
    vc = (pc[:, 1] / zc).astype(np.int32)

    flat, zz, rr = _collect_splat(uc, vc, zc.astype(np.float32), rgb, H, W, splat)
    if flat.size == 0:
        return clr

    uf, _, rgbf = _zbuffer_first(flat, zz, rr)
    clr.reshape(-1, 3)[uf] = rgbf
    return clr



def project_depth_sem_clr(
    pts_xyz: np.ndarray, pts_sem: np.ndarray,
    clr_xyz: np.ndarray, clr_rgb: np.ndarray,
    image_from_lidar: np.ndarray, ori_hw,
    *, invalid_depth=-300.0, splat=None, n_actor_classes=3
):
    """
    pts_xyz: (N,3) lidar
    pts_sem: (N,) int, 0=bg/ignore, 1..K=classes
    clr_xyz: (M,3) lidar
    clr_rgb: (M,3) uint8
    return:
      depth: (H,W) float32
      sem  : (H,W,K) uint8(0/1)
      clr  : (H,W,3) uint8
    """
    H, W = int(ori_hw[0]), int(ori_hw[1])
    if splat is None:
        splat = [(1e9, 0)]

    # ----- depth -----
    xyz1 = np.concatenate([pts_xyz.astype(np.float32), np.ones((pts_xyz.shape[0], 1), np.float32)], 1)
    p = xyz1 @ image_from_lidar.T
    z = p[:, 2]
    m = z > 1e-5
    p, z = p[m], z[m]
    sem_id = pts_sem[m].astype(np.int32) if pts_sem is not None else None

    u = (p[:, 0] / z).astype(np.int32)
    v = (p[:, 1] / z).astype(np.int32)

    depth = np.full((H, W), np.inf, np.float32)
    z2, u2, v2 = z.copy(), u.copy(), v.copy()
    for zmax, r in splat:
        mm = z2 <= zmax
        if not np.any(mm):
            continue
        uu, vv, zz = _expand_uv(u2[mm], v2[mm], z2[mm], r, H, W)
        np.minimum.at(depth.reshape(-1), vv * W + uu, zz)
        z2, u2, v2 = z2[~mm], u2[~mm], v2[~mm]
        if z2.size == 0:
            break
    depth[~np.isfinite(depth)] = float(invalid_depth)

    # ----- sem (z-buffer) -----
    
    sem = np.zeros((H, W, int(n_actor_classes)), np.uint8)
    if sem_id is not None and sem_id.size > 0:
        good = (sem_id >= 1) & (sem_id <= n_actor_classes)
        zc, uc, vc, sid = z[good], u[good], v[good], sem_id[good].astype(np.int32)
        flat, zz, sid2 = _collect_splat(uc, vc, zc, sid, H, W, splat)
        if flat.size > 0:
            uf, _, sf = _zbuffer_first(flat, zz, sid2.astype(np.int32))
            sf = sf - 1
            ok = (sf >= 0) & (sf < n_actor_classes)
            sem.reshape(-1, n_actor_classes)[uf[ok], sf[ok]] = 1
        
    # ----- clr (z-buffer) -----
    clr = np.zeros((H, W, 3), np.uint8)
    if clr_xyz is not None and clr_xyz.shape[0] > 0:
        xyz1c = np.concatenate([clr_xyz.astype(np.float32), np.ones((clr_xyz.shape[0], 1), np.float32)], 1)
        pc = xyz1c @ image_from_lidar.T
        zc = pc[:, 2]
        mc = zc > 1e-5
        pc, zc, rgb = pc[mc], zc[mc], clr_rgb[mc].astype(np.uint8)
        uc = (pc[:, 0] / zc).astype(np.int32)
        vc = (pc[:, 1] / zc).astype(np.int32)

        flat, zz, rr = _collect_splat(uc, vc, zc, rgb, H, W, splat)
        if flat.size > 0:
            uf, _, rgbf = _zbuffer_first(flat, zz, rr)
            clr.reshape(-1, 3)[uf] = rgbf

    return depth, sem, clr

############ tools ##################


def _zbuffer_first(flat: np.ndarray, z: np.ndarray, payload: np.ndarray | None):
    """
    return: uniq_flat, uniq_z, uniq_payload
    """
    order = np.lexsort((z, flat))
    flat2 = flat[order]
    z2 = z[order]
    first = np.r_[True, flat2[1:] != flat2[:-1]]
    uf = flat2[first]
    uz = z2[first]
    if payload is None:
        return uf, uz, None
    p2 = payload[order]
    return uf, uz, p2[first]


def _collect_splat(u, v, z, payload, H, W, splat):
    """
    splat: list[(zmax, r)]
    """
    flat_all, z_all, p_all = [], [], []
    u = u.astype(np.int32); v = v.astype(np.int32); z = z.astype(np.float32)

    for zmax, r in splat:
        mm = z <= zmax
        if not np.any(mm):
            continue

        uu = u[mm]; vv = v[mm]; zz = z[mm]
        pp = payload[mm] if payload is not None else None

        if r > 0:
            d = np.arange(-r, r + 1, dtype=np.int32)
            dx, dy = np.meshgrid(d, d)
            dx = dx.reshape(-1); dy = dy.reshape(-1)
            k2 = dx.size

            uu = (uu[:, None] + dx[None, :]).reshape(-1)
            vv = (vv[:, None] + dy[None, :]).reshape(-1)
            zz = np.repeat(zz, k2)
            if pp is not None:
                pp = np.repeat(pp, k2, axis=0) if pp.ndim == 2 else np.repeat(pp, k2)

        m_in = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
        uu = uu[m_in]; vv = vv[m_in]; zz = zz[m_in]
        flat_all.append(vv * W + uu)
        z_all.append(zz)
        if pp is not None:
            p_all.append(pp[m_in])

        keep = ~mm
        u = u[keep]; v = v[keep]; z = z[keep]
        if payload is not None:
            payload = payload[keep]
        if z.size == 0:
            break

    flat = np.concatenate(flat_all, 0) if flat_all else np.zeros((0,), np.int32)
    z = np.concatenate(z_all, 0) if z_all else np.zeros((0,), np.float32)
    p = np.concatenate(p_all, 0) if (payload is not None and p_all) else None
    return flat, z, p



# ============================================================
# ============================================================

def parse_val_cfg(cfg: dict) -> dict:
    """
    """
    d0 = cfg["validation_dataset"]["base_dataset"]["datasets"][0]
    transforms = cfg["validation_dataset"]["transform_list"]

    # 1) seq / cams
    seq_len = int(d0.get("sequence_length", 1))
    sensor_channels = list(d0.get("sensor_channels", []))

    # 2) resize to (H,W) from transform_list
    out_hw = None
    for tr in transforms:
        if tr.get("new_key") in ("3dbox_images", "hdmap_images", "vae_images"):
            comp = tr.get("transform", {})
            ts = comp.get("transforms", [])
            for t in ts:
                if t.get("_class_name", "").endswith("Resize"):
                    size = t.get("size", None)
                    if size is not None and len(size) == 2:
                        out_hw = (int(size[0]), int(size[1]))  # (H,W)
                        break
        if out_hw is not None:
            break

    if out_hw is None:
        out_hw = (448, 896)

    # 3) text settings
    text_settings = d0.get("image_description_settings", {}) or {}
    text_path = text_settings.get("path", None)
    reorder_keys = bool(text_settings.get("reorder_keys", False))
    align_keys = text_settings.get("align_keys", ["time", "weather"])

    return {
        "sequence_length": seq_len,
        "sensor_channels": sensor_channels,
        "out_hw": out_hw,  # (H,W)
        "text_path": text_path,
        "text_align_keys": align_keys,
        "text_reorder_keys": reorder_keys,
        "stub_key_data_dict": d0.get("stub_key_data_dict", {}) or {},
    }




def parse_val_proj_cfg(cfg: dict) -> dict | None:
    """
    """
    d0 = cfg["validation_dataset"]["base_dataset"]["datasets"][0]
    s = d0.get("projected_pc_settings", None)
    if not s:
        return None

    out = dict(s)

    if "n_actor_classes" not in out:
        out["n_actor_classes"] = 3
    if "depth_bins" not in out:
        out["depth_bins"] = 256
    if "depth_bin_mode" not in out:
        out["depth_bin_mode"] = "log"
    if "log_gamma" not in out:
        out["log_gamma"] = 1.0
    if "invalid_depth" not in out:
        out["invalid_depth"] = -300.0
    if "splat" not in out:
        out["splat"] = [(1e9, 0)]

    dt = str(out.get("data_type", "")).lower().strip()
    if dt not in ("depth", "clr", "all"):
        raise ValueError(f"projected_pc_settings['data_type'] must be depth/clr/all, got {dt}")

    return out


class ProjContext:
    """
    """
    def __init__(self, proj_cfg: dict):
        self.cfg = proj_cfg

        # --- bg scene ---
        self.bg_scene = {}
        cs = proj_cfg.get("color_scene_by_location", {}) or {}
        for k, npy_path in cs.items():
            self.bg_scene[k] = np.asarray(np.load(npy_path, allow_pickle=False))

        # --- actor template ---
        self.actor_root = proj_cfg.get("actor_root", None)
        tpl_root = proj_cfg.get("actor_template_root", None)
        self.actor_tpl = {}

        if tpl_root:
            for fn in os.listdir(tpl_root):
                if fn.endswith(".pkl"):
                    with open(os.path.join(tpl_root, fn), "rb") as f:
                        self.actor_tpl[fn[:-4]] = pickle.load(f)


# ===========================
# [ADD] compose points (bg + actors), same as your Dataset
# ===========================

def yaw_to_Rz(yaw: float) -> np.ndarray:
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float32)


def global_to_ego(xyz_g: np.ndarray, R_ego2glb: np.ndarray, t_ego_glb: np.ndarray) -> np.ndarray:
    # xyz_g: (N,3), R_ego2glb: (3,3), t: (3,)
    return (xyz_g - t_ego_glb[None, :]) @ R_ego2glb


def as_list(x):
    if x is None:
        return []
    return list(x) if isinstance(x, (list, tuple, np.ndarray)) else [x]


def pick_car_template(actor_tpl: dict, dz: float):
    dz = float(dz)
    if dz < 1.8:
        return actor_tpl.get("sedan")
    if dz < 2.05:
        return actor_tpl.get("suv")
    if dz < 2.7:
        return actor_tpl.get("pickup")
    return None


def compose_points_for_frame(info: dict, ctx: ProjContext):
    """
      pts_xyz (N,3) float32  (ego/lidar frame)
      pts_sem (N,)  int32    0=bg, 1..K=actors
      clr_xyz (M,3) float32  (ego/lidar frame)
      clr_rgb (M,3) uint8
    """
    s = ctx.cfg
    radius = float(s.get("radius", 150.0))
    min_actor_pts = int(s.get("min_actor_points", 20000))

    ego2global = np.asarray(info["ego2global"], np.float32)
    R = ego2global[:3, :3]   # ego->global
    t = ego2global[:3, 3]

    # ---------- 1) background ----------
    bg = ctx.bg_scene.get(info.get("db_name"), None)
    if bg is None:
        bg_xyz_l = np.zeros((0, 3), np.float32)
        bg_rgb = np.zeros((0, 3), np.uint8)
    else:
        bg = np.asarray(bg)
        xyz_g = bg[:, :3].astype(np.float32)
        rgb = np.clip(bg[:, 3:6], 0, 255).astype(np.uint8)

        m = (xyz_g[:, 0] > t[0] - radius) & (xyz_g[:, 0] < t[0] + radius) & \
            (xyz_g[:, 1] > t[1] - radius) & (xyz_g[:, 1] < t[1] + radius)
        xyz_g, rgb = xyz_g[m], rgb[m]

        bg_xyz_l = global_to_ego(xyz_g, R, t).astype(np.float32)
        bg_rgb = rgb

    bg_sem = np.zeros((bg_xyz_l.shape[0],), np.int32)

    # ---------- 2) actors ----------
    boxes = info.get("gt_boxes", None)
    if boxes is None:
        boxes = []
    boxes = np.asarray(boxes, np.float32)

    names = as_list(info.get("gt_names", None))
    tracks = info.get("track_token", None)
    tracks = as_list(tracks)
    if len(tracks) == 0:
        tracks = [None] * len(boxes)

    name2id = {"car": 1, "ped": 2, "bike": 3}

    act_xyz_list, act_sem_list = [], []
    act_clr_xyz_list, act_clr_rgb_list = [], []

    for i in range(len(boxes)):
        box = boxes[i]
        x, y, z, dx, dy, dz, yaw = box[:7].astype(np.float32)

        cls = str(names[i]) if i < len(names) else ""
        sid = name2id.get(cls, 0)
        if sid == 0:
            continue

        tok = tracks[i] if i < len(tracks) else None

        actor_track = None
        if tok is not None and ctx.actor_root is not None:
            p = os.path.join(ctx.actor_root, f"{tok}.npy")
            if os.path.isfile(p):
                actor_track = np.asarray(np.load(p))
                if actor_track.shape[0] == 0:
                    actor_track = None

        actor_sem = None
        if actor_track is not None and actor_track.shape[0] > min_actor_pts:
            actor_sem = actor_track
        else:
            if cls == "car":
                actor_sem = pick_car_template(ctx.actor_tpl, dz)
            elif cls == "ped":
                actor_sem = ctx.actor_tpl.get("ped")
            elif cls == "bike":
                actor_sem = ctx.actor_tpl.get("bike")

        if actor_sem is None:
            continue

        actor_sem = np.asarray(actor_sem)
        xyz_sem = actor_sem[:, :3].astype(np.float32)

        Rz = yaw_to_Rz(yaw)
        xyz_sem = (Rz @ xyz_sem.T).T + np.array([x, y, z], np.float32)[None, :]

        act_xyz_list.append(xyz_sem)
        act_sem_list.append(np.full((xyz_sem.shape[0],), sid, np.int32))

        if actor_track is None or actor_track.shape[1] < 6:
            continue

        xyz_clr = actor_track[:, :3].astype(np.float32)
        xyz_clr = (Rz @ xyz_clr.T).T + np.array([x, y, z], np.float32)[None, :]

        a_rgb = actor_track[:, 3:6]
        if a_rgb.size == 0:
            continue

        mx = np.nanmax(a_rgb)
        if mx <= 1.5:
            a_rgb_u8 = np.clip(a_rgb * 255.0, 0, 255).astype(np.uint8)
        else:
            a_rgb_u8 = np.clip(a_rgb, 0, 255).astype(np.uint8)

        act_clr_xyz_list.append(xyz_clr)
        act_clr_rgb_list.append(a_rgb_u8)

    act_xyz = np.concatenate(act_xyz_list, 0) if act_xyz_list else np.zeros((0, 3), np.float32)
    act_sem = np.concatenate(act_sem_list, 0) if act_sem_list else np.zeros((0,), np.int32)
    act_clr_xyz = np.concatenate(act_clr_xyz_list, 0) if act_clr_xyz_list else np.zeros((0, 3), np.float32)
    act_clr_rgb = np.concatenate(act_clr_rgb_list, 0) if act_clr_rgb_list else np.zeros((0, 3), np.uint8)

    pts_xyz = np.concatenate([bg_xyz_l, act_xyz], 0).astype(np.float32)
    pts_sem = np.concatenate([bg_sem, act_sem], 0).astype(np.int32)

    clr_xyz = np.concatenate([bg_xyz_l, act_clr_xyz], 0).astype(np.float32)
    clr_rgb = np.concatenate([bg_rgb, act_clr_rgb], 0).astype(np.uint8)

    return pts_xyz, pts_sem, clr_xyz, clr_rgb



def vis_from_bins_cache(bins_u16_cache: np.ndarray, n_bins: int) -> Image.Image:
    bins_u16_cache = np.asarray(bins_u16_cache, np.uint16)
    vis_bgr = visualize_bins_u16(
        bins_u16_cache,
        n_bins=int(n_bins),
        invalid_bin=0,
        colormap=cv2.COLORMAP_TURBO,
    )
    vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(vis_rgb, "RGB")


def compute_proj_for_sequence(
    cfg: dict,
    seq_infos: list,
    sensor_channels: list,
    *,
    cam_key="cam",
) -> dict:
    """
      proj_depth: [T,V,H,W] torch.long (bins)
      proj_sem  : [T,V,K,H,W] torch.float32
      proj_clr  : [T,V,3,H,W] torch.float32
    """
    proj_cfg = parse_val_proj_cfg(cfg)
    if proj_cfg is None:
        return {"proj_depth": None, "proj_sem": None, "proj_clr": None}

    ctx = ProjContext(proj_cfg)

    final_hw = proj_cfg.get("final_hw", None)  # [H,W]
    if final_hw is None:
        raise ValueError("projected_pc_settings['final_hw'] must exist (cfg has it).")
    final_hw = (int(final_hw[0]), int(final_hw[1]))

    invalid = float(proj_cfg["invalid_depth"])
    splat = proj_cfg["splat"]
    n_bins = int(proj_cfg["depth_bins"])
    gamma = float(proj_cfg["log_gamma"])
    far_m = float(proj_cfg.get("radius", 50.0))
    mode = str(proj_cfg.get("depth_bin_mode", "log")).lower()
    n_cls = int(proj_cfg.get("n_actor_classes", 3))

    dt = str(proj_cfg["data_type"]).lower().strip()
    want_depth = (dt in ("depth", "all"))
    want_clr = (dt in ("clr", "all"))
    want_sem = True

    need_vis_depth = want_depth

    T = len(seq_infos)
    V = len(sensor_channels)

    out_depth = [] if want_depth else None
    out_sem = [] if want_sem else None
    out_clr = [] if want_clr else None
    out_vis = [] if need_vis_depth else None

    for t in range(T):
        info = seq_infos[t]

        pts_xyz, pts_sem, clr_xyz, clr_rgb = compose_points_for_frame(info, ctx)

        row_d = [] if want_depth else None
        row_s = [] if want_sem else None
        row_c = [] if want_clr else None
        row_v = [] if need_vis_depth else None

        cam_pack = info.get(cam_key, {}) or {}

        for v in range(V):
            ch = sensor_channels[v]
            cam_info = cam_pack.get(ch, None)
            if cam_info is None:
                H0, W0 = final_hw
                if want_depth:
                    row_d.append(torch.zeros((H0, W0), dtype=torch.long))
                if want_sem:
                    row_s.append(torch.zeros((n_cls, H0, W0), dtype=torch.float32))
                if want_clr:
                    row_c.append(torch.zeros((3, H0, W0), dtype=torch.float32))
                if need_vis_depth:
                    row_v.append(torch.zeros((3, H0, W0), dtype=torch.float32))
                continue

            # ori_hw
            ori_hw_cfg = proj_cfg.get("ori_hw", [1080, 1920])
            ori_hw = (int(ori_hw_cfg[0]), int(ori_hw_cfg[1]))  # (H,W)

            # lidar->image
            l2i = lidar2image_from_caminfo(cam_info)  

            depth = None
            sem = None
            clr = None

            if want_depth:
                depth = project_depth_only(
                    pts_xyz, l2i, ori_hw,
                    invalid_depth=invalid, splat=splat
                )

            if want_sem:
                sem = project_sem_only(
                    pts_xyz, pts_sem, l2i, ori_hw,
                    splat=splat, n_actor_classes=n_cls
                )

            if want_clr:
                clr = project_clr_only(
                    clr_xyz, clr_rgb, l2i, ori_hw,
                    splat=splat
                )

            if want_depth and depth is not None:
                depth = downsample_depth_blockwise(depth, final_hw, invalid=invalid)

            if want_sem and sem is not None:
                # sem: (H,W,K) -> resize nearest
                sem_u8 = (sem * 255).astype(np.uint8)
                sem_u8 = cv2.resize(
                    sem_u8, (final_hw[1], final_hw[0]),
                    interpolation=cv2.INTER_NEAREST
                )
                sem = (sem_u8 > 127).astype(np.uint8)

            if want_clr and clr is not None:
                clr = downsample_clr_blockwise(clr, final_hw)

            H0, W0 = final_hw

            if want_depth:
                if mode in ("linear", "lin", "abs"):
                    bins_u16 = depth_to_linbins_u16(depth, invalid=invalid, n_bins=n_bins, far_m=far_m)
                else:
                    bins_u16 = depth_to_logbins_u16(depth, invalid=invalid, n_bins=n_bins, far_m=far_m, gamma=gamma)

                bins_u16 = np.asarray(bins_u16, np.uint16)
                row_d.append(torch.from_numpy(bins_u16.astype(np.int64)))

                vis_img = vis_from_bins_cache(bins_u16, n_bins)
                vis_np = np.asarray(vis_img, np.uint8).copy()
                row_v.append(torch.from_numpy(vis_np).permute(2, 0, 1).float() / 255.)

            if want_sem:
                sem_np = sem.astype(np.float32)  # (H,W,K) 0/1
                row_s.append(torch.from_numpy(sem_np).permute(2, 0, 1))  # (K,H,W)

            if want_clr:
                clr_np = clr.astype(np.uint8)
                row_c.append(torch.from_numpy(clr_np).permute(2, 0, 1).float() / 255.)

        if want_depth:
            out_depth.append(torch.stack(row_d, 0))  # [V,H,W]
        if want_sem:
            out_sem.append(torch.stack(row_s, 0))    # [V,K,H,W]
        if want_clr:
            out_clr.append(torch.stack(row_c, 0))    # [V,3,H,W]
        if need_vis_depth:
            out_vis.append(torch.stack(row_v, 0))

    ret = {}
    if want_depth:
        ret["proj_depth"] = torch.stack(out_depth, 0)  # [T,V,H,W]
    else:
        ret["proj_depth"] = None

    ret["proj_sem"] = torch.stack(out_sem, 0) if want_sem else None  # [T,V,K,H,W]
    ret["proj_clr"] = torch.stack(out_clr, 0) if want_clr else None  # [T,V,3,H,W]

    if need_vis_depth:
        ret["vis_depth"] = torch.stack(out_vis, 0)

    return ret


# ============================================================
# ============================================================

def se3_from_qt_wxyz(q_wxyz, t_xyz) -> np.ndarray:
    q_wxyz = np.asarray(q_wxyz, np.float32).reshape(4)
    t_xyz = np.asarray(t_xyz, np.float32).reshape(3)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = Quaternion(q_wxyz).rotation_matrix.astype(np.float32)
    T[:3, 3] = t_xyz
    return T


def se3_inv(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, np.float32).reshape(4, 4)
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float32)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def lidar2image_from_caminfo(cam_info: dict) -> np.ndarray:
    """
    """
    K3 = np.asarray(cam_info["camera_intrinsics"], np.float32).reshape(3, 3)

    R_c2l = np.asarray(cam_info["sensor2lidar_rotation"], np.float32).reshape(3, 3)
    t_c2l = np.asarray(cam_info["sensor2lidar_translation"], np.float32).reshape(3)

    R_l2c = R_c2l.T
    t_l2c = -R_l2c @ t_c2l

    Rt = np.concatenate([R_l2c, t_l2c[:, None]], axis=1)  # 3x4
    P = K3 @ Rt

    l2i = np.eye(4, dtype=np.float32)
    l2i[:3, :4] = P
    return l2i


def build_camera_intrinsics_pad4(seq_infos: list, sensor_channels: list) -> np.ndarray:
    """
    """
    T = len(seq_infos)
    V = len(sensor_channels)
    out = np.zeros((T, V, 4, 4), np.float32)

    for t in range(T):
        info = seq_infos[t]
        cam = info.get("cam", {}) or {}
        for v, ch in enumerate(sensor_channels):
            ci = cam.get(ch, None)
            K_pad = np.eye(4, dtype=np.float32)
            if ci is not None:
                K3 = np.asarray(ci["camera_intrinsics"], np.float32).reshape(3, 3)
                K_pad[:3, :3] = K3
            out[t, v] = K_pad
    return out


def build_camera_transforms(seq_infos: list, sensor_channels: list) -> np.ndarray:
    """
    camera_transforms: ego_from_camera  [T,V,4,4]
    """
    T = len(seq_infos)
    V = len(sensor_channels)
    out = np.zeros((T, V, 4, 4), np.float32)

    for t in range(T):
        info = seq_infos[t]
        cam = info.get("cam", {}) or {}
        for v, ch in enumerate(sensor_channels):
            ci = cam.get(ch, None)
            if ci is None:
                out[t, v] = np.eye(4, dtype=np.float32)
            else:
                out[t, v] = se3_from_qt_wxyz(ci["sensor2ego_rotation"], ci["sensor2ego_translation"])
    return out


def build_ego_transforms(seq_infos: list, sensor_channels: list) -> np.ndarray:
    """
    ego_transforms: world_from_ego  [T,V,4,4]
    """
    T = len(seq_infos)
    V = len(sensor_channels)
    out = np.zeros((T, V, 4, 4), np.float32)

    for t in range(T):
        ego2global = np.asarray(seq_infos[t].get("ego2global", np.eye(4)), np.float32).reshape(4, 4)
        for v in range(V):
            out[t, v] = ego2global
    return out


# ============================================================
# ============================================================

def load_text_anno(text_json_path: str | None) -> dict:
    if not text_json_path:
        return {}
    with open(text_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_clip_text_for_scene(seq_infos: list, text_anno: dict, text_cfg: dict) -> str:
    """
      cap = text_anno.get(db_name,{})
      tail = "This is a nuplan video clip from {location}"
    """
    if len(seq_infos) == 0:
        return "This is a nuplan video clip"

    info0 = seq_infos[0]
    db_name = info0.get("db_name", None)
    location = info0.get("location", "")

    cap = text_anno.get(db_name, {}) if db_name is not None else {}
    align_keys = text_cfg.get("align_keys", ["time", "weather"])
    reorder = bool(text_cfg.get("reorder_keys", False))

    keys = list(align_keys)
    if reorder and len(keys) > 1:
        keys = keys[::-1]

    parts = []
    for k in keys:
        if k in cap:
            parts.append(str(cap[k]))

    text_main = ". ".join(parts).strip()
    tail = f"This is a nuplan video clip from {location}" if location else "This is a nuplan video clip"
    if text_main:
        return f"{text_main}. {tail}"
    return tail


# ============================================================
# ============================================================

_BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (0, 4), (1, 5), (2, 6), (3, 7),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 5), (0, 7)
]



def box_corners_lidar_xyz(box7: np.ndarray, z_is_center: bool = True) -> np.ndarray:
    box7 = np.asarray(box7, np.float32).reshape(7)
    x, y, z, dx, dy, dz, yaw = box7.tolist()

    xs = np.array([ dx/2,  dx/2, -dx/2, -dx/2,  dx/2,  dx/2, -dx/2, -dx/2], np.float32)
    ys = np.array([ dy/2, -dy/2, -dy/2,  dy/2,  dy/2, -dy/2, -dy/2,  dy/2], np.float32)

    if z_is_center:
        zs = np.array([-dz/2, -dz/2, -dz/2, -dz/2, dz/2, dz/2, dz/2, dz/2], np.float32)
    else:
        zs = np.array([0, 0, 0, 0, dz, dz, dz, dz], np.float32)

    R = yaw_to_Rz(yaw)
    pts = np.stack([xs, ys, zs], axis=0)
    pts = (R @ pts).T
    pts += np.array([x, y, z], np.float32)
    return pts


def project_points_xyz(pts_xyz: np.ndarray, T_4x4: np.ndarray) -> np.ndarray:
    pts_xyz = np.asarray(pts_xyz, np.float32)
    T_4x4 = np.asarray(T_4x4, np.float32).reshape(4, 4)

    pts_h = np.concatenate([pts_xyz, np.ones((pts_xyz.shape[0], 1), np.float32)], axis=1)
    q = pts_h @ T_4x4.T
    z = q[:, 2]
    z_safe = np.clip(z, 1e-5, 1e9)
    u = q[:, 0] / z_safe
    v = q[:, 1] / z_safe
    return np.stack([u, v, z], axis=1)


def render_3dbox_one_view(
    info: dict,
    cam_ch: str,
    base_wh=(1920, 1080),
    out_hw=(448, 896),
    pen_width: int = 8,
    color_table=None,
) -> Image.Image:
    """
    """
    if color_table is None:
        color_table = {"ped": (255, 0, 0), "bike": (128, 255, 0), "car": (0, 0, 255)}

    W0, H0 = int(base_wh[0]), int(base_wh[1])
    img = Image.new("RGB", (W0, H0))
    draw = ImageDraw.Draw(img)

    cam_info = (info.get("cam", {}) or {}).get(cam_ch, None)
    if cam_info is None:
        return img.resize((int(out_hw[1]), int(out_hw[0])), Image.BILINEAR)

    l2i = lidar2image_from_caminfo(cam_info)

    boxes = info.get("gt_boxes", None)
    if boxes is None:
        return img.resize((int(out_hw[1]), int(out_hw[0])), Image.BILINEAR)

    names = info.get("gt_names", [])
    names = list(names) if isinstance(names, (list, tuple, np.ndarray)) else []

    boxes = np.asarray(boxes, np.float32)
    for i in range(boxes.shape[0]):
        cls = str(names[i]) if i < len(names) else ""
        color = color_table.get(cls, (0, 0, 0))

        corners = box_corners_lidar_xyz(boxes[i, :7], z_is_center=True)
        uvz = project_points_xyz(corners, l2i)

        if np.any(uvz[:, 2] <= 1e-5):
            continue

        pts2 = uvz[:, :2]
        if (pts2[:, 0].max() < 0) or (pts2[:, 0].min() > W0) or (pts2[:, 1].max() < 0) or (pts2[:, 1].min() > H0):
            continue

        for a, b in _BOX_EDGES:
            xa, ya = pts2[a]
            xb, yb = pts2[b]
            draw.line((float(xa), float(ya), float(xb), float(yb)), fill=tuple(color), width=int(pen_width))

    # resize to cfg out_hw
    return img.resize((int(out_hw[1]), int(out_hw[0])), Image.BILINEAR)


def render_3dbox_sequence(seq_infos: list, sensor_channels: list, out_hw) -> list:
    """
    return: nested list [T][V] PIL.Image
    """
    T = len(seq_infos)
    out = []
    for t in range(T):
        row = []
        for ch in sensor_channels:
            row.append(render_3dbox_one_view(seq_infos[t], ch, base_wh=(1920, 1080), out_hw=out_hw))
        out.append(row)
    return out


# ============================================================
# ============================================================

def render_hdmap_one_view(
    map_api,
    info: dict,
    cam_ch: str,
    base_wh=(1920, 1080),
    out_hw=(448, 896),
    pen_width: int = 8,
    patch_radius: float = 100.0,
    near_plane: float = 1e-8,
) -> Image.Image:
    """

    """
    W0, H0 = int(base_wh[0]), int(base_wh[1])
    canvas = np.zeros((H0, W0, 3), np.uint8)

    cam_info = (info.get("cam", {}) or {}).get(cam_ch, None)
    if cam_info is None:
        return Image.fromarray(canvas[..., ::-1]).resize((int(out_hw[1]), int(out_hw[0])), Image.BILINEAR)

    ego2global = np.asarray(info.get("ego2global", None), np.float32)
    if ego2global is None or ego2global.shape != (4, 4):
        return Image.fromarray(canvas[..., ::-1]).resize((int(out_hw[1]), int(out_hw[0])), Image.BILINEAR)

    # BGR for cv2
    blue = (255, 0, 0)
    green = (0, 255, 0)
    red = (0, 0, 255)

    from nuplan.common.actor_state.state_representation import Point2D
    center = Point2D(float(ego2global[0, 3]), float(ego2global[1, 3]))

    from nuplan.common.maps.maps_datatypes import SemanticMapLayer

    polygon_layers = [
        SemanticMapLayer.LANE,
        SemanticMapLayer.INTERSECTION,
        SemanticMapLayer.WALKWAYS,
        SemanticMapLayer.CARPARK_AREA,
        SemanticMapLayer.CROSSWALK,
    ]
    line_layers = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]

    ego_R = ego2global[:3, :3]
    ego_t = ego2global[:3, 3].copy()
    ego_t[2] = 0.0

    T_s2e = se3_from_qt_wxyz(cam_info["sensor2ego_rotation"], cam_info["sensor2ego_translation"])
    T_e2s = se3_inv(T_s2e)

    K3 = np.asarray(cam_info["camera_intrinsics"], np.float32).reshape(3, 3)

    def worldxy_to_uv(world_xy: np.ndarray, is_polygon: bool):
        if world_xy.shape[0] < (3 if is_polygon else 2):
            return None

        pts_w3 = np.concatenate([world_xy.astype(np.float32), np.zeros((world_xy.shape[0], 1), np.float32)], axis=1)
        pts_ego = (pts_w3 - ego_t[None, :]) @ ego_R
        pts_ego_h = np.concatenate([pts_ego, np.ones((pts_ego.shape[0], 1), np.float32)], axis=1)
        pts_cam_h = pts_ego_h @ T_e2s.T
        pts_cam = pts_cam_h[:, :3].T  # 3xN

        if np.all(pts_cam[2, :] < near_plane):
            return None

        m = pts_cam[2, :] > near_plane
        pts_cam = pts_cam[:, m]
        if pts_cam.shape[1] < (3 if is_polygon else 2):
            return None

        uvw = K3 @ pts_cam
        z = np.clip(uvw[2, :], 1e-5, 1e9)
        u = uvw[0, :] / z
        v = uvw[1, :] / z

        uv = np.stack([u, v], axis=1)
        return uv

    try:
        polys = map_api.get_proximal_map_objects(center, patch_radius, polygon_layers)
        drivable_layers = {
            SemanticMapLayer.LANE,
            SemanticMapLayer.INTERSECTION,
            SemanticMapLayer.WALKWAYS,
            SemanticMapLayer.CARPARK_AREA,
        }
        for ln, objs in polys.items():
            if ln not in drivable_layers:
                continue
            for o in objs:
                poly = getattr(o, "polygon", None)
                if poly is None or poly.is_empty:
                    continue
                ext = np.asarray(poly.exterior.coords, np.float32)
                uv = worldxy_to_uv(ext[:, :2], is_polygon=True)
                if uv is None:
                    continue
                pts = np.round(uv).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(canvas, [pts], isClosed=True, color=blue, thickness=int(pen_width))
    except Exception:
        pass

    try:
        lines = map_api.get_proximal_map_objects(center, patch_radius, line_layers)
        for _, objs in lines.items():
            for lane in objs:
                for path in (lane.left_boundary.discrete_path, lane.right_boundary.discrete_path):
                    pts = [(float(p.x), float(p.y)) for p in path]
                    if len(pts) < 2:
                        continue
                    pts = np.asarray(pts, np.float32)
                    uv = worldxy_to_uv(pts, is_polygon=False)
                    if uv is None:
                        continue
                    seg = np.round(uv).astype(np.int32).reshape(-1, 1, 2)
                    cv2.polylines(canvas, [seg], isClosed=False, color=green, thickness=max(1, int(pen_width // 2)))
    except Exception:
        pass

    try:
        polys = map_api.get_proximal_map_objects(center, patch_radius, [SemanticMapLayer.CROSSWALK])
        for o in polys.get(SemanticMapLayer.CROSSWALK, []):
            poly = getattr(o, "polygon", None)
            if poly is None or poly.is_empty:
                continue
            ext = np.asarray(poly.exterior.coords, np.float32)
            uv = worldxy_to_uv(ext[:, :2], is_polygon=True)
            if uv is None:
                continue
            pts = np.round(uv).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], isClosed=True, color=red, thickness=int(pen_width))
    except Exception:
        pass

    img = Image.fromarray(canvas[..., ::-1])
    return img.resize((int(out_hw[1]), int(out_hw[0])), Image.BILINEAR)


def render_hdmap_sequence(map_api, seq_infos: list, sensor_channels: list, out_hw) -> list:
    """
    return: nested list [T][V] PIL.Image
    """
    T = len(seq_infos)
    out = []
    for t in range(T):
        row = []
        for ch in sensor_channels:
            row.append(render_hdmap_one_view(map_api, seq_infos[t], ch, base_wh=(1920, 1080), out_hw=out_hw))
        out.append(row)
    return out


# ============================================================
# ============================================================

def build_conditions_by_cfg(
    cfg: dict,
    seq_infos: list,
    *,
    map_api=None,
    want_proj: bool = False,
    pts_xyz=None,
    pts_sem=None,
    clr_xyz=None,
    clr_rgb=None,
) -> dict:
    """
    """
    s = parse_val_cfg(cfg)

    T_need = int(s["sequence_length"])
    if len(seq_infos) >= T_need:
        seq = seq_infos[:T_need]
    else:
        seq = list(seq_infos)
        while len(seq) < T_need:
            seq.append(seq[-1])
    sensor_channels = s["sensor_channels"]
    out_hw = s["out_hw"]

    # ---- matrices ----
    camera_intrinsics = build_camera_intrinsics_pad4(seq, sensor_channels)   # [T,V,4,4]
    camera_transforms = build_camera_transforms(seq, sensor_channels)        # [T,V,4,4]
    ego_transforms = build_ego_transforms(seq, sensor_channels)              # [T,V,4,4]

    # ---- conditions images ----
    box_imgs = render_3dbox_sequence(seq, sensor_channels, out_hw=out_hw)

    if map_api is None:
        hd_imgs = [[Image.new("RGB", (out_hw[1], out_hw[0])) for _ in sensor_channels] for _ in range(len(seq))]
    else:
        hd_imgs = render_hdmap_sequence(map_api, seq, sensor_channels, out_hw=out_hw)

    # ---- clip_text ----
    text_anno = load_text_anno(s["text_path"])
    clip_text = make_clip_text_for_scene(
        seq_infos=seq,
        text_anno=text_anno,
        text_cfg={"align_keys": s["text_align_keys"], "reorder_keys": s["text_reorder_keys"]},
    )
    clip_text_TV = [[clip_text for _ in sensor_channels] for _ in range(len(seq))]  # [T][V]

    # ---- stub crossview_mask ----
    stub = s["stub_key_data_dict"]
    crossview_mask = None
    if "crossview_mask" in stub:
        payload = stub["crossview_mask"][1]
        data = payload.get("data", {})
        if data.get("_class_name", "") == "json.loads":
            crossview_mask = json.loads(data["s"])

    out = {
        "camera_intrinsics": camera_intrinsics,
        "camera_transforms": camera_transforms,
        "ego_transforms": ego_transforms,
        "3dbox_images": box_imgs,          # [T][V] PIL.Image (448x896)
        "hdmap_images": hd_imgs,           # [T][V] PIL.Image (448x896)
        "clip_text": clip_text_TV,         # [T][V] str
        "crossview_mask": crossview_mask,  # python list[list[int]] or None
    }

    if want_proj:
        proj_ret = compute_proj_for_sequence(cfg, seq, sensor_channels)

        if proj_ret.get("proj_depth", None) is not None:
            out["proj_depth"] = proj_ret["proj_depth"]
            out["vis_depth"] = proj_ret["vis_depth"]

        if proj_ret.get("proj_sem", None) is not None:
            out["proj_sem"] = proj_ret["proj_sem"]

        if proj_ret.get("proj_clr", None) is not None:
            out["proj_clr"] = proj_ret["proj_clr"]

    return out
