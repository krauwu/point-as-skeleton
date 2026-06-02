import os, json, time, pickle
import numpy as np
import torch
import cv2

import dwm.common
import dwm.datasets.common

# choose splits from info_pkl
# import dwm.datasets.nuplan_splits

from PIL import Image, ImageDraw, ImageFile
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points
from shapely.ops import unary_union
from shapely.geometry import LineString, MultiLineString, Polygon

from nuplan.common.maps.nuplan_map.map_factory import NuPlanMapFactory, get_maps_db
from nuplan.database.nuplan_db.nuplan_db_utils import get_lidarpc_sensor_data
from nuplan.database.nuplan_db.nuplan_scenario_queries import get_sensor_token_map_name_from_db
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, StopLineType
from nuplan.common.actor_state.state_representation import Point2D
import os, cv2, numpy as np, torch
from PIL import Image

from dwm.datasets.common import (
    _safe_save_png, _try_open_png,
    _safe_save_u16_png, _try_open_u16_png,
    _acquire_lock, _release_lock,
    ensure_cache_subdir,
    depth_to_logbins_u16, depth_to_linbins_u16,
    visualize_bins_u16,
    downsample_depth_blockwise, downsample_clr_blockwise,
)

cv2.setNumThreads(0)

# -------- cache decode: u16png -> bin_id --------
def u16cache_to_binid(u16: np.ndarray, n_bins: int) -> np.ndarray:
    if u16 is None:
        return None
    u16 = np.asarray(u16)
    if u16.size == 0:
        return u16.astype(np.int32)
    if u16.max() <= n_bins - 1:
        return u16.astype(np.int32)
    scale = max(1, 65535 // int(n_bins))
    return (u16.astype(np.int32) // scale).clip(0, n_bins - 1)

# -------- splat expand --------
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
        
    # sem = np.zeros((H, W, int(n_actor_classes)), np.uint8)
    # if sem_id is not None and sem_id.size > 0:
    #     flat_all, z_all, sid_all = [], [], []
    #     zc, uc, vc, sid = z, u, v, sem_id
    #     good = (sid >= 1) & (sid <= n_actor_classes)
    #     zc, uc, vc, sid = zc[good], uc[good], vc[good], sid[good]
    #     for zmax, r in splat:
    #         mm = zc <= zmax
    #         if not np.any(mm):
    #             continue
    #         uu, vv, zz = _expand_uv(uc[mm], vc[mm], zc[mm], r, H, W)
    #         ss = np.repeat(sid[mm], (2 * r + 1) * (2 * r + 1)) if r > 0 else sid[mm].copy()

    #         if r > 0:
    #             ss = ss[: uu.shape[0]]

    #         flat_all.append(vv * W + uu)
    #         z_all.append(zz)
    #         sid_all.append(ss.astype(np.int32))

    #         zc, uc, vc, sid = zc[~mm], uc[~mm], vc[~mm], sid[~mm]
    #         if zc.size == 0:
    #             break

    #     if flat_all:
    #         flat = np.concatenate(flat_all, 0)
    #         zz   = np.concatenate(z_all, 0)
    #         sid  = np.concatenate(sid_all, 0)

    #         order = np.lexsort((zz, flat))
    #         flat, sid = flat[order], sid[order]
    #         first = np.r_[True, flat[1:] != flat[:-1]]
    #         uf = flat[first]
    #         sf = sid[first] - 1  # -> 0..K-1
    #         sem.reshape(-1, n_actor_classes)[uf, sf] = 1

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
    
    # clr = np.zeros((H, W, 3), np.uint8)
    # if clr_xyz is not None and clr_xyz.shape[0] > 0:
    #     xyz1c = np.concatenate([clr_xyz.astype(np.float32), np.ones((clr_xyz.shape[0], 1), np.float32)], 1)
    #     pc = xyz1c @ image_from_lidar.T
    #     zc = pc[:, 2]
    #     mc = zc > 1e-5
    #     pc, zc, rgb = pc[mc], zc[mc], clr_rgb[mc].astype(np.uint8)
    #     uc = (pc[:, 0] / zc).astype(np.int32)
    #     vc = (pc[:, 1] / zc).astype(np.int32)

    #     flat_all, z_all, rgb_all = [], [], []
    #     zt, ut, vt, rt = zc, uc, vc, rgb
    #     for zmax, r in splat:
    #         mm = zt <= zmax
    #         if not np.any(mm):
    #             continue

    #         if r <= 0:
    #             uu, vv, zz = _expand_uv(ut[mm], vt[mm], zt[mm], 0, H, W)
    #             rr = rt[mm]
    #         else:
    #             d = np.arange(-r, r + 1, dtype=np.int32)
    #             dx, dy = np.meshgrid(d, d)
    #             dx = dx.reshape(-1); dy = dy.reshape(-1)
    #             k2 = dx.size

    #             uu0 = (ut[mm][:, None] + dx[None, :]).reshape(-1)
    #             vv0 = (vt[mm][:, None] + dy[None, :]).reshape(-1)
    #             zz0 = np.repeat(zt[mm], k2)
    #             rr0 = np.repeat(rt[mm], k2, axis=0)

    #             m_in = (uu0 >= 0) & (uu0 < W) & (vv0 >= 0) & (vv0 < H)
    #             uu, vv, zz, rr = uu0[m_in], vv0[m_in], zz0[m_in], rr0[m_in]

    #         flat_all.append(vv * W + uu)
    #         z_all.append(zz)
    #         rgb_all.append(rr)

    #         zt, ut, vt, rt = zt[~mm], ut[~mm], vt[~mm], rt[~mm]
    #         if zt.size == 0:
    #             break

    #     if flat_all:
    #         flat = np.concatenate(flat_all, 0)
    #         zz   = np.concatenate(z_all, 0)
    #         rr   = np.concatenate(rgb_all, 0)

    #         order = np.lexsort((zz, flat))
    #         flat, rr = flat[order], rr[order]
    #         first = np.r_[True, flat[1:] != flat[:-1]]
    #         clr.reshape(-1, 3)[flat[first]] = rr[first]

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

def layer_to_key(layer_name):
    if layer_name in (SemanticMapLayer.CARPARK_AREA, SemanticMapLayer.WALKWAYS, SemanticMapLayer.INTERSECTION):
        return "drivable_area"
    if layer_name == SemanticMapLayer.LANE:
        return "lane"
    if layer_name == SemanticMapLayer.CROSSWALK:
        return "ped_crossing"
    if layer_name == SemanticMapLayer.STOP_LINE:
        return "lane"  
    return None

def _find_nearest(sorted_arr, x):
    i = int(np.searchsorted(sorted_arr, x))
    if i <= 0:
        return 0
    if i >= len(sorted_arr):
        return len(sorted_arr) - 1
    return i - 1 if abs(sorted_arr[i - 1] - x) <= abs(sorted_arr[i] - x) else i


def make_image_description_string(caption_dict: dict, settings: dict, random_state=None):
    settings = settings or {}
    rng = random_state if random_state is not None else np.random.default_rng()

    keys = ["time", "weather"]
    if settings.get("reorder_keys", False):
        keys = [keys[i] for i in rng.permutation(len(keys))]

    dr = settings.get("drop_rates")
    if dr:
        keys = [k for k in keys if not (k in dr and rng.random() <= dr[k])]

    return ". ".join(str(caption_dict[k]) for k in keys if k in caption_dict)

_BOX_EDGES = [
    # bottom face (0-1-2-3)
    (0, 1), (1, 2), (2, 3), (3, 0),
    # verticals
    (0, 4), (1, 5), (2, 6), (3, 7),
    # top face (4-5-6-7)
    (4, 5), (5, 6), (6, 7), (7, 4),
    # extra "direction" lines (keep your style)
    (6, 3), (3, 4)
]

def _yaw_to_Rz(yaw):
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float32)


def _box_corners_lidar_xyz(box7, z_is_center=False):
    # box7: [x,y,z, dx,dy,dz, yaw] in LiDAR frame
    x, y, z, dx, dy, dz, yaw = box7[:7].astype(np.float32)
    xs = np.array([ dx/2,  dx/2, -dx/2, -dx/2,  dx/2,  dx/2, -dx/2, -dx/2], np.float32)
    ys = np.array([ dy/2, -dy/2, -dy/2,  dy/2,  dy/2, -dy/2, -dy/2,  dy/2], np.float32)
    if z_is_center:
        zs = np.array([ -dz/2, -dz/2, -dz/2, -dz/2,  dz/2,  dz/2,  dz/2,  dz/2], np.float32)
    else:
        zs = np.array([ 0, 0, 0, 0, dz, dz, dz, dz], np.float32)

    R = _yaw_to_Rz(yaw)
    pts = np.stack([xs, ys, zs], axis=0)    # (3,8)
    pts = (R @ pts).T                       # (8,3)
    pts += np.array([x, y, z], np.float32)
    return pts


def _project_pts(pts_xyz, T_4x4):
    pts_h = np.concatenate([pts_xyz.astype(np.float32),
                            np.ones((pts_xyz.shape[0], 1), np.float32)], axis=1)
    q = pts_h @ T_4x4.T
    z = np.clip(q[:, 2], 1e-5, 1e9)
    u = q[:, 0] / z
    v = q[:, 1] / z
    return np.stack([u, v, q[:, 2]], axis=1)


def _lidar2image_from_caminfo(cam_info, K3):
    # cam_info: contains sensor2lidar_rotation/translation
    R_c2l = np.asarray(cam_info["sensor2lidar_rotation"], np.float32).reshape(3, 3)
    t_c2l = np.asarray(cam_info["sensor2lidar_translation"], np.float32).reshape(3)

    # invert: lidar -> cam
    R_l2c = R_c2l.T
    t_l2c = -R_l2c @ t_c2l

    Rt = np.concatenate([R_l2c, t_l2c[:, None]], axis=1)      # 3x4
    P  = np.asarray(K3, np.float32).reshape(3, 3) @ Rt        # 3x4

    l2i = np.eye(4, dtype=np.float32)
    l2i[:3, :4] = P
    return l2i


def _infer_TV(result: dict):
    v = result.get("vae_images", None)
    if torch.is_tensor(v) and v.ndim >= 2:
        return int(v.shape[0]), int(v.shape[1])
    if isinstance(v, (list, tuple)):
        T = len(v)
        V = max((len(x) for x in v), default=0)
        return T, V

    c = result.get("camera_intrinsics", None)
    if torch.is_tensor(c) and c.ndim >= 2:
        return int(c.shape[0]), int(c.shape[1])

    p = result.get("pts", None)
    if torch.is_tensor(p) and p.ndim >= 1:
        return int(p.shape[0]), 0

    return 1, 0

def _se3_from_qt(q, t):
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = Quaternion(q).rotation_matrix.astype(np.float32)
    T[:3, 3] = np.asarray(t, np.float32).reshape(3)
    return T

def _se3_inv(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float32)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti

############ tools ##################

class NuPlanDataset(torch.utils.data.Dataset):
    """
    """

    default_3dbox_color_table = {
        "ped": (255, 0, 0),
        "bike": (128, 255, 0),
        "car": (0, 0, 255),
    }
    default_hdmap_color_table = {
        "drivable_area": (0, 0, 255),
        "lane": (0, 255, 0),
        "ped_crossing": (255, 0, 0)
    }


    def __init__(
        self,
        sensor_root,
        pkl_path,
        cache_root,
        dataset_root,                 # dataset_root/{db_name}.db
        map_root,
        map_version='nuplan-maps-v1.0',
        sequence_length=1,
        fps_stride_tuples=(),
        sensor_channels=('CAM_L1','CAM_L0','CAM_F0','CAM_R0','CAM_R1','CAM_R2','CAM_B0','CAM_L2'),
        scene_key="db_name",
        timestamp_key="timestamp",
        token_key="token",
        enable_synchronization_check=True,
        max_time_error_ratio=0.5,
        max_time_error_us=None,
        enable_sample_data=False,
        # render
        _3dbox_pen_width=8,
        hdmap_pen_width=8,
        hdmap_patch_radius=100.0,
        near_plane=1e-8,
        min_polygon_area=2000,
        # cam calib
        cam_key="cam",
        # ego pose
        ego2global_key="ego2global",         # 4x4
        # text & subkey
        text_anno='./data/nuplan/nuplan_scene.json',
        stub_key_data_dict=None,
        image_description_settings=None,
        # pts
        projected_pc_settings=None
    ):
        self.sensor_root = sensor_root
        self.cache_root = cache_root

        self.dataset_root = dataset_root
        self.map_factory = NuPlanMapFactory(get_maps_db(map_root=map_root, map_version=map_version))

        self.sequence_length = int(sequence_length)
        self.fps_stride_tuples = list(fps_stride_tuples)
        self.sensor_channels = list(sensor_channels)
        self.scene_key = scene_key
        self.timestamp_key = timestamp_key
        self.token_key = token_key
        self.enable_synchronization_check = bool(enable_synchronization_check)
        self.max_time_error_ratio = float(max_time_error_ratio)
        self.max_time_error_us = None if max_time_error_us is None else int(max_time_error_us)
        self.enable_sample_data = bool(enable_sample_data)

        self._3dbox_pen_width = int(_3dbox_pen_width)
        self.hdmap_pen_width = int(hdmap_pen_width)
        self.hdmap_patch_radius = float(hdmap_patch_radius)
        self.near_plane = float(near_plane)
        self.min_polygon_area = float(min_polygon_area)

        self.cam_key = cam_key
        self.ego2global_key = ego2global_key
        self.stub_key_data_dict = stub_key_data_dict
        
        self.projected_pc_settings = projected_pc_settings
        self.do_cache = bool(projected_pc_settings and projected_pc_settings.get("do_cache", True))

        # nuplan layers
        self.polygon_layer_names = [
            SemanticMapLayer.LANE,
            SemanticMapLayer.CROSSWALK,
            SemanticMapLayer.INTERSECTION,
            SemanticMapLayer.STOP_LINE,
            SemanticMapLayer.WALKWAYS,
            SemanticMapLayer.CARPARK_AREA,
        ]
        self.line_layer_names = [
            SemanticMapLayer.LANE,
            SemanticMapLayer.LANE_CONNECTOR,
        ]

        infos = self._load_infos(pkl_path)
        self.scenes, self.scene_ts = self._build_scenes(infos)
        self.items = self._build_items()
        
        self._map_cache = {}        # map_name -> numap
        self._token2map = {}        # (log_db_path, lidar_token) -> map_name
        self._hdgeom_cache = {}     # (log_db_path, lidar_token) -> precomputed world-geometry
        self._hdgeom_cache_max = 4096
        
        with open(text_anno, "r", encoding="utf-8") as f:
            self.text_anno = json.load(f)
        self.text_settings = image_description_settings
        
        # pts_proj
        
        self._bg_scene = {}
        if projected_pc_settings and "color_scene_by_location" in projected_pc_settings:
            self._bg_scene = {k: np.load(v, allow_pickle=False)
                for k, v in projected_pc_settings["color_scene_by_location"].items()}

        # actor root / templates 
        self._actor_root = projected_pc_settings.get("actor_root", None) if projected_pc_settings else None
        self._actor_tpl = {}
        tpl_root = projected_pc_settings.get("actor_template_root", None) if projected_pc_settings else None
        if tpl_root and not self._actor_tpl:
            for fn in os.listdir(tpl_root):
                if fn.endswith(".pkl"):
                    with open(os.path.join(tpl_root, fn), "rb") as f:
                        self._actor_tpl[fn[:-4]] = pickle.load(f)
            
        ensure_cache_subdir(self.cache_root, "3dbox_images")
        ensure_cache_subdir(self.cache_root, "hdmap_images")
        
    def _png_path(self, subdir, token):
        return os.path.join(self.cache_root, subdir, f"{token}.png")

    @staticmethod
    def _load_infos(pkl_path):
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)
        infos = obj["infos"] if isinstance(obj, dict) and "infos" in obj else obj
        if not isinstance(infos, list):
            raise TypeError("PKL must be list[dict] or {'infos': list[dict]}")
        return infos
    # ---------------------------
    # clip indices
    # ---------------------------
    def _build_scenes(self, infos):
        scenes = {}
        for info in infos:
            scene = info.get(self.scene_key)
            t = info.get(self.timestamp_key)
            if scene is None or t is None:
                continue
            scenes.setdefault(scene, []).append(info)

        scene_ts = {}
        for s, lst in scenes.items():
            lst.sort(key=lambda x: x[self.timestamp_key])
            scene_ts[s] = np.asarray([x[self.timestamp_key] for x in lst], dtype=np.int64)

        scenes = {s: lst for s, lst in scenes.items() if len(lst) >= self.sequence_length}
        scene_ts = {s: ts for s, ts in scene_ts.items() if len(ts) >= self.sequence_length}
        return scenes, scene_ts

    def _build_items(self):
        items = []
        for scene, ts in self.scene_ts.items():
            for fps, stride in self.fps_stride_tuples:
                items.extend(self._enumerate_segments(scene, ts, fps, stride))
        return items

    def _enumerate_segments(self, scene, ts, fps, stride):
        T, N = self.sequence_length, len(ts)
        items = []

        if float(fps) == 0.0:
            s = max(1, int(stride))
            for start in range(0, N - T + 1, s):
                items.append({"scene": scene, "fps": 0.0, "indices": list(range(start, start + T))})
            return items

        fps = float(fps)
        dt_us = int(round(1e6 / fps))
        seq_dur_us = (T - 1) * dt_us

        t_begin = int(ts[0])
        t_last_begin = int(ts[-1] - seq_dur_us)
        if t_last_begin < t_begin:
            return items

        stride_sec = float(stride)
        stride_us = dt_us if stride_sec <= 0 else int(round(stride_sec * 1e6))

        max_err_us = self.max_time_error_us
        if max_err_us is None:
            max_err_us = int(self.max_time_error_ratio * dt_us)

        t = t_begin
        while t <= t_last_begin:
            wanted = np.array([t + i * dt_us for i in range(T)], dtype=np.int64)
            idxs = [_find_nearest(ts, w) for w in wanted]
            if len(set(idxs)) != T:
                t += stride_us
                continue

            if self.enable_synchronization_check:
                picked = ts[np.asarray(idxs, dtype=np.int64)]
                if int(np.max(np.abs(picked - wanted))) > max_err_us:
                    t += stride_us
                    continue

            items.append({"scene": scene, "fps": fps, "indices": idxs})
            t += stride_us

        return items

    # ---------------------------
    # paths + calib
    # ---------------------------
    def _get_sensor_paths(self, info):
        img = info.get('img_filename') or []
        m = {p.split("/")[-2]: p for p in img}
        return [os.path.join(self.sensor_root, m.get(ch, "")) for ch in self.sensor_channels]

    def _get_cam_info(self, info, cam_ch):
        cam = info.get(self.cam_key) or {}
        return cam.get(cam_ch)

    def _img_token(self, info, cam_idx):
        tok = info.get(self.token_key, None)
        if tok is None:
            tok = info.get(self.timestamp_key, "t")
        return f"{tok}_{cam_idx}"

    # ---------------------------
    # hdmap projection (nuplan)
    # ---------------------------
    @staticmethod
    def _clip_points_behind_camera(points, near_plane: float, is_polygon: bool):
        pts = []
        assert points.shape[0] == 3
        n = points.shape[1]
        m = n if is_polygon else n - 1
        for i1 in range(m):
            i2 = (i1 + 1) % n
            p1, p2 = points[:, i1], points[:, i2]
            z1, z2 = p1[2], p2[2]
            if z1 >= near_plane and z2 >= near_plane:
                if len(pts) == 0 or np.any(pts[-1] != p1):
                    pts.append(p1)
                pts.append(p2)
            elif z1 < near_plane and z2 < near_plane:
                continue
            else:
                if z1 <= z2:
                    pa, pb = p1, p2
                else:
                    pa, pb = p2, p1
                za, zb = pa[2], pb[2]
                d = pb - pa
                alpha = (near_plane - zb) / (za - zb)
                clipped = pa + (1 - alpha) * d
                if z1 >= near_plane and (len(pts) == 0 or np.any(pts[-1] != p1)):
                    pts.append(p1)
                pts.append(clipped)
                if z2 >= near_plane:
                    pts.append(p2)
        return np.array(pts).transpose() if len(pts) else np.zeros((3, 0), np.float32)

    def _perspective_coords(self, points, im_size, ego2global_t, ego2global_r, sensor2ego_t, sensor2ego_r, cam_intrinsic, is_polygon):
        # keep your current "flatten z" behavior exactly
        ego2global_t = np.concatenate([ego2global_t[:2], np.zeros_like(ego2global_t)[:1]], axis=-1)
        points = points - np.array(ego2global_t).reshape((-1, 1))
        points = np.dot(ego2global_r.T, points)

        points = points - np.array(sensor2ego_t).reshape((-1, 1))
        points = np.dot(Quaternion(sensor2ego_r).rotation_matrix.T, points)

        depths = points[2, :]
        if np.all(depths < self.near_plane):
            return None

        points = self._clip_points_behind_camera(points, self.near_plane, is_polygon)
        if is_polygon and (points.size == 0 or points.shape[1] < 3):
            return None

        points = view_points(points, cam_intrinsic, normalize=True)

        inside = np.ones(points.shape[1], dtype=bool)
        inside &= points[0, :] > 1
        inside &= points[0, :] < im_size[0] - 1
        inside &= points[1, :] > 1
        inside &= points[1, :] < im_size[1] - 1
        if np.all(~inside):
            return None
        return points

    @staticmethod
    def _draw_linestring_rgb(canvas_rgb, line: MultiLineString, color_bgr, thickness):
        def ic(x): return np.array(x).round().astype(np.int32)
        segs = [ic(ls.coords).reshape(-1, 1, 2) for ls in line.geoms]
        if len(segs):
            cv2.polylines(canvas_rgb, segs, isClosed=False, color=color_bgr, thickness=int(thickness))

    def _get_numap(self, log_db_path, token):
        k = (log_db_path, token)
        if k not in self._token2map:
            self._token2map[k] = get_sensor_token_map_name_from_db(
                log_db_path, get_lidarpc_sensor_data(), token
            )
        name = self._token2map[k]
        if name not in self._map_cache:
            self._map_cache[name] = self.map_factory.build_map_from_name(name)
        return self._map_cache[name]

    def _get_hdmap_world_geom(self, info):
        token = info["lidarpc_token"]
        db_name = info[self.scene_key]
        db_path = db_name if str(db_name).endswith(".db") else str(db_name) + ".db"
        log_db_path = os.path.join(self.dataset_root, db_path)
        key = (log_db_path, token)

        hit = self._hdgeom_cache.get(key, None)
        if hit is not None:
            return hit

        ego2global = np.asarray(info[self.ego2global_key], np.float32)
        center = Point2D(float(ego2global[0, 3]), float(ego2global[1, 3]))

        numap = self._get_numap(log_db_path, token)

        # ---- polygons
        nearest_poly = numap.get_proximal_map_objects(center, self.hdmap_patch_radius, self.polygon_layer_names)
        if SemanticMapLayer.STOP_LINE in nearest_poly:
            nearest_poly[SemanticMapLayer.STOP_LINE] = [
                sp for sp in nearest_poly[SemanticMapLayer.STOP_LINE]
                if sp.stop_line_type != StopLineType.TURN_STOP
            ]

        # 1) drivable union exteriors (world xy)
        drivable_layers = {
            SemanticMapLayer.LANE,
            SemanticMapLayer.INTERSECTION,
            SemanticMapLayer.WALKWAYS,
            SemanticMapLayer.CARPARK_AREA,
        }
        drivable_polys = []
        for ln, objs in nearest_poly.items():
            if ln not in drivable_layers:
                continue
            for o in objs:
                poly = getattr(o, "polygon", None)
                if poly is not None and (not poly.is_empty):
                    drivable_polys.append(poly)

        drivable_exteriors = []
        if drivable_polys:
            uni = unary_union(drivable_polys)
            geoms = uni.geoms if uni.geom_type == "MultiPolygon" else [uni]
            for g in geoms:
                ext = np.asarray(g.exterior.coords, np.float32)
                if ext.shape[0] >= 2:
                    drivable_exteriors.append(ext)

        # 2) divider lines (world xy) via seg count
        nearest_line = numap.get_proximal_map_objects(center, self.hdmap_patch_radius, self.line_layer_names)

        def _seg_key(p, q, quant=0.2):
            p = tuple(np.round(np.asarray(p) / quant).astype(np.int32))
            q = tuple(np.round(np.asarray(q) / quant).astype(np.int32))
            return (p, q) if p <= q else (q, p)

        seg_cnt = {}
        for _, objs in nearest_line.items():
            for lane in objs:
                for path in (lane.left_boundary.discrete_path, lane.right_boundary.discrete_path):
                    pts = [(float(p.x), float(p.y)) for p in path]
                    for i in range(len(pts) - 1):
                        k = _seg_key(pts[i], pts[i + 1])
                        seg_cnt[k] = seg_cnt.get(k, 0) + 1

        divider_lines = []
        for _, objs in nearest_line.items():
            for lane in objs:
                for path in (lane.left_boundary.discrete_path, lane.right_boundary.discrete_path):
                    pts = [(float(p.x), float(p.y)) for p in path]
                    if len(pts) < 2:
                        continue
                    flags = [seg_cnt.get(_seg_key(pts[i], pts[i + 1]), 0) >= 2 for i in range(len(pts) - 1)]
                    if (sum(flags) / max(1, len(flags))) < 0.6:
                        continue
                    divider_lines.append(np.asarray(pts, np.float32))

        # 3) crosswalk exteriors (world xy)
        crosswalk_exteriors = []
        for obj in nearest_poly.get(SemanticMapLayer.CROSSWALK, []):
            poly = getattr(obj, "polygon", None)
            if poly is None or poly.is_empty:
                continue
            ext = np.asarray(poly.exterior.coords, np.float32)
            if ext.shape[0] >= 3:
                crosswalk_exteriors.append(ext)

        out = {
            "drivable_exteriors": drivable_exteriors,
            "divider_lines": divider_lines,
            "crosswalk_exteriors": crosswalk_exteriors,
        }

        if len(self._hdgeom_cache) >= self._hdgeom_cache_max:
            self._hdgeom_cache.pop(next(iter(self._hdgeom_cache)))
        self._hdgeom_cache[key] = out
        return out

    # ---------- _get_hdmap_image ----------
    def _get_hdmap_image(self, info, cam_ch, im_size, cam_intrinsic_3x3):
        W, H = map(int, im_size)
        canvas = np.zeros((H, W, 3), np.uint8)

        ego2global = np.asarray(info[self.ego2global_key], np.float32)
        ego2global_t, ego2global_r = ego2global[:3, 3], ego2global[:3, :3]

        cam_info = self._get_cam_info(info, cam_ch)
        if cam_info is None:
            return Image.fromarray(canvas[..., ::-1])

        sensor2ego_t = np.asarray(cam_info["sensor2ego_translation"], np.float32)
        sensor2ego_r = cam_info["sensor2ego_rotation"]

        # cached heavy geometry (per token)
        geom = self._get_hdmap_world_geom(info)

        # inner clip polygon (same as your code)
        margin = 2.0
        scene_poly = Polygon([(margin, margin), (margin, H - margin), (W - margin, H - margin), (W - margin, margin)])

        # colors (same as your code)
        blue_rgb = (0, 0, 255)
        blue_bgr = (blue_rgb[2], blue_rgb[1], blue_rgb[0])

        green_rgb = (0, 255, 0)
        green_bgr = (green_rgb[2], green_rgb[1], green_rgb[0])

        red_rgb = (255, 0, 0)
        red_bgr = (red_rgb[2], red_rgb[1], red_rgb[0])

        # --- 1) blue: drivable union exteriors
        for ext_xy in geom["drivable_exteriors"]:
            pts3 = np.vstack([ext_xy.T, np.zeros((1, ext_xy.shape[0]), np.float32)])  # 3xN
            q = self._perspective_coords(
                pts3, (W, H),
                ego2global_t, ego2global_r,
                sensor2ego_t, sensor2ego_r,
                cam_intrinsic_3x3,
                is_polygon=True
            )
            if q is None:
                continue
            xy_img = np.stack([q[0], q[1]], axis=1).astype(np.float32)
            line = LineString([(float(x), float(y)) for x, y in xy_img]).intersection(scene_poly)
            if line.is_empty:
                continue
            if line.geom_type == "LineString":
                line = MultiLineString([line])
            self._draw_linestring_rgb(canvas, line, blue_bgr, thickness=int(self.hdmap_pen_width))

        # --- 2) green: divider lines (already filtered in world)
        for pts in geom["divider_lines"]:
            pts2 = pts.T.astype(np.float32)  # 2xN
            pts3 = np.vstack([pts2, np.zeros((1, pts2.shape[1]), np.float32)])
            q = self._perspective_coords(
                pts3, (W, H),
                ego2global_t, ego2global_r,
                sensor2ego_t, sensor2ego_r,
                cam_intrinsic_3x3,
                is_polygon=False
            )
            if q is None:
                continue
            xy_img = np.stack([q[0], q[1]], axis=1).astype(np.float32)
            line = LineString([(float(x), float(y)) for x, y in xy_img]).intersection(scene_poly)
            if line.is_empty:
                continue
            if line.geom_type == "LineString":
                line = MultiLineString([line])
            self._draw_linestring_rgb(canvas, line, green_bgr, thickness=max(1, int(self.hdmap_pen_width // 2)))

        # --- 3) red: crosswalk exteriors
        for ext_xy in geom["crosswalk_exteriors"]:
            pts3 = np.vstack([ext_xy.T, np.zeros((1, ext_xy.shape[0]), np.float32)])  # 3xN
            q = self._perspective_coords(
                pts3, (W, H),
                ego2global_t, ego2global_r,
                sensor2ego_t, sensor2ego_r,
                cam_intrinsic_3x3,
                is_polygon=True
            )
            if q is None:
                continue
            xy_img = np.stack([q[0], q[1]], axis=1).astype(np.float32)
            line = LineString([(float(x), float(y)) for x, y in xy_img]).intersection(scene_poly)
            if line.is_empty:
                continue

            if line.geom_type == "LineString":
                line = MultiLineString([line])
            elif line.geom_type == "GeometryCollection":
                parts = [g for g in line.geoms if g.geom_type in ("LineString", "MultiLineString")]
                if not parts:
                    continue
                line = unary_union(parts)
                if line.geom_type == "LineString":
                    line = MultiLineString([line])
                elif line.geom_type != "MultiLineString":
                    continue
            elif line.geom_type != "MultiLineString":
                continue

            self._draw_linestring_rgb(canvas, line, red_bgr, thickness=int(self.hdmap_pen_width))

        return Image.fromarray(canvas[..., ::-1])

    # ---------------------------
    # 3dbox from gt_line
    # ---------------------------
    def _get_3dbox_image_from_gtline(self, info, im_size, lidar2image_4x4):
        W, H = int(im_size[0]), int(im_size[1])
        img = Image.new("RGB", (W, H))
        draw = ImageDraw.Draw(img)

        boxes = info.get("gt_line")
        names = info.get('gt_names')
        names = list(names) if isinstance(names, (list, tuple, np.ndarray)) else None

        for i, box in enumerate(boxes):
            color = (0, 0, 0)
            if names is not None and i < len(names):
                color = self.default_3dbox_color_table.get(str(names[i]), color)

            corners = _box_corners_lidar_xyz(box[:7], z_is_center=False)
            uvz = _project_pts(corners, lidar2image_4x4)
            if np.any(uvz[:, 2] <= 1e-5):
                continue

            pts2 = uvz[:, :2]
            if (pts2[:, 0].max() < 0) or (pts2[:, 0].min() > W) or (pts2[:, 1].max() < 0) or (pts2[:, 1].min() > H):
                continue

            for a, b in _BOX_EDGES:
                xa, ya = pts2[a]
                xb, yb = pts2[b]
                draw.line((float(xa), float(ya), float(xb), float(yb)),
                          fill=tuple(color), width=self._3dbox_pen_width)

        return img

    # ---------------------------
    # pts_proj
    # ---------------------------
    
    @staticmethod
    def _as_list(x):
        if x is None:
            return []
        return list(x) if isinstance(x, (list, tuple, np.ndarray)) else [x]

    @staticmethod
    def _global_to_ego(xyz_g: np.ndarray, R_ego2glb: np.ndarray, t_ego_glb: np.ndarray):
        # xyz_g: (N,3), R: (3,3), t: (3,)
        return (xyz_g - t_ego_glb[None, :]) @ R_ego2glb

    def _pick_car_template(self, dz: float):
        dz = float(dz)
        if dz < 1.8:  return self._actor_tpl.get("sedan")
        if dz < 2.05: return self._actor_tpl.get("suv")
        if dz < 2.7:  return self._actor_tpl.get("pickup")
        return None

    @staticmethod
    def _seg_key(p, q, quant=0.2):
        p = tuple(np.round(np.asarray(p) / quant).astype(np.int32))
        q = tuple(np.round(np.asarray(q) / quant).astype(np.int32))
        return (p, q) if p <= q else (q, p)
        
    def _compose_points_for_frame(self, info):
        s = self.projected_pc_settings or {}
        radius = float(s.get("radius", 150.0))
        min_actor_pts = int(s.get("min_actor_points", 20000))

        ego2global = np.asarray(info[self.ego2global_key], np.float32)
        R = ego2global[:3, :3]   # ego->global 
        t = ego2global[:3, 3]

        # ---------- 1) background ----------
        bg = self._bg_scene.get(info.get(self.scene_key), None)
        if bg is None:
            bg_xyz_l = np.zeros((0, 3), np.float32)
            bg_rgb   = np.zeros((0, 3), np.uint8)
        else:
            bg = np.asarray(bg)
            xyz_g = bg[:, :3].astype(np.float32)
            rgb   = np.clip(bg[:, 3:6], 0, 255).astype(np.uint8)

            m = (xyz_g[:, 0] > t[0] - radius) & (xyz_g[:, 0] < t[0] + radius) & \
                (xyz_g[:, 1] > t[1] - radius) & (xyz_g[:, 1] < t[1] + radius)
            xyz_g, rgb = xyz_g[m], rgb[m]

            bg_xyz_l = self._global_to_ego(xyz_g, R, t).astype(np.float32)
            bg_rgb   = rgb

        bg_sem = np.zeros((bg_xyz_l.shape[0],), np.int32)

        # ---------- 2) actors ----------
        boxes = info.get("gt_boxes", None)
        if boxes is None:
            boxes = []
        names  = self._as_list(info.get("gt_names", None))
        tracks = info.get("track_token", None)
        if tracks is None:
            tracks = info.get("gt_track_token", None)
        tracks = self._as_list(tracks)
        if len(tracks) == 0:
            tracks = [None] * len(boxes)

        name2id = {"car": 1, "ped": 2, "bike": 3}

        act_xyz_list, act_sem_list = [], []
        act_clr_xyz_list, act_clr_rgb_list = [], []

        for i, box in enumerate(boxes):
            x, y, z, dx, dy, dz, yaw = box[:7].astype(np.float32)

            cls = str(names[i]) if i < len(names) else ""
            sid = name2id.get(cls, 0)
            if sid == 0:
                continue

            tok = tracks[i] if i < len(tracks) else None

            actor_track = None
            if tok is not None and self._actor_root is not None:
                p = os.path.join(self._actor_root, f"{tok}.npy")
                if os.path.isfile(p):
                    actor_track = np.asarray(np.load(p))

                    if actor_track.shape[0] == 0:
                        actor_track = None

            actor_sem = None
            if actor_track is not None and actor_track.shape[0] > min_actor_pts:
                actor_sem = actor_track
            else:
                if cls == "car":
                    actor_sem = self._pick_car_template(dz)
                elif cls == "ped":
                    actor_sem = self._actor_tpl.get("ped")
                elif cls == "bike":
                    actor_sem = self._actor_tpl.get("bike")
            if actor_sem is None:
                continue

            actor_sem = np.asarray(actor_sem)
            xyz_sem = actor_sem[:, :3].astype(np.float32)
            Rz = _yaw_to_Rz(yaw)
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
            a_rgb_u8 = (np.clip(a_rgb * 255.0, 0, 255).astype(np.uint8)
                        if mx <= 1.5 else np.clip(a_rgb, 0, 255).astype(np.uint8))

            act_clr_xyz_list.append(xyz_clr)
            act_clr_rgb_list.append(a_rgb_u8)

        act_xyz = np.concatenate(act_xyz_list, 0) if act_xyz_list else np.zeros((0, 3), np.float32)
        act_sem = np.concatenate(act_sem_list, 0) if act_sem_list else np.zeros((0,), np.int32)
        act_clr_xyz = np.concatenate(act_clr_xyz_list, 0) if act_clr_xyz_list else np.zeros((0, 3), np.float32)
        act_clr_rgb = np.concatenate(act_clr_rgb_list, 0) if act_clr_rgb_list else np.zeros((0, 3), np.uint8)

        pts_xyz = np.concatenate([bg_xyz_l, act_xyz], 0).astype(np.float32)
        pts_sem = np.concatenate([bg_sem,   act_sem], 0).astype(np.int32)
        clr_xyz = np.concatenate([bg_xyz_l, act_clr_xyz], 0).astype(np.float32)
        clr_rgb = np.concatenate([bg_rgb,   act_clr_rgb], 0).astype(np.uint8)

        return pts_xyz, pts_sem, clr_xyz, clr_rgb
    
    # ---------------------------
    # main api
    # ---------------------------
    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        scene, idxs, fps = item["scene"], item["indices"], item["fps"]

        seq = [self.scenes[scene][i] for i in idxs]
        location = seq[0]['location']
        t0 = seq[0][self.timestamp_key]
        pts = torch.tensor([(x[self.timestamp_key] - t0 + 500) // 1000 for x in seq], dtype=torch.float32)

        result = {"fps": torch.tensor(float(fps), dtype=torch.float32), "pts": pts}

        if self.enable_sample_data:
            result["sample_data"] = seq
            result["scene"] = {"token": scene}

        cam_infos = []
        
        # ---- images: undistort + intrinsics
        images, cam_Ks, img_sizes, dists = [], [], [], []
        for x in seq:
            paths = self._get_sensor_paths(x)
            row_imgs, row_K, row_sz, row_dist = [], [], [], []
            row_infos = []  

            for cam_i, (cam_ch, p) in enumerate(zip(self.sensor_channels, paths)):
                im = _try_open_png(p)
                cam_info = self._get_cam_info(x, cam_ch)
                row_infos.append(cam_info)

                if im is None or cam_info is None:
                    W, H = 1920, 1080
                    row_imgs.append(Image.new("RGB", (W, H)))
                    K_pad = np.eye(4, dtype=np.float32)
                    row_K.append(K_pad)
                    row_sz.append((W, H))
                    row_dist.append(np.zeros((0,), np.float32))
                    continue
                
                row_imgs.append(im)

                K3 = np.asarray(cam_info["camera_intrinsics"], np.float32).reshape(3, 3)
                K_pad = np.eye(4, dtype=np.float32)
                K_pad[:3, :3] = K3
                row_K.append(K_pad)

                row_sz.append(im.size)  # (W,H)

                dist = np.asarray(cam_info.get("distortion", []), np.float32).reshape(-1)
                row_dist.append(dist)
                
            cam_infos.append(row_infos)
            images.append(row_imgs)
            cam_Ks.append(row_K)
            img_sizes.append(row_sz)
            dists.append(row_dist)

        result["images"] = images
        result["camera_intrinsics"] = torch.tensor(np.asarray(cam_Ks), dtype=torch.float32)
        result["image_size"] = torch.tensor(
            np.asarray([[[w, h] for (w, h) in row] for row in img_sizes], dtype=np.int64),
            dtype=torch.long
        )
        # result["distortion"] = dists

        # ---- 3dbox_images (gt_line) cached
        cached = []
        all_hit = True
        for t, x in enumerate(seq):
            row = []
            for v, _ in enumerate(self.sensor_channels):
                tok = self._img_token(x, v)
                p = self._png_path("3dbox_images", tok)
                img = _try_open_png(p) if os.path.isfile(p) else None
                if img is None:
                    all_hit = False
                row.append(img)
            cached.append(row)

        if all_hit:
            result["3dbox_images"] = cached
        else:
            imgs = []
            for t, x in enumerate(seq):
                row = []
                for v, cam_ch in enumerate(self.sensor_channels):
                    W, H = img_sizes[t][v]

                    cam_info = self._get_cam_info(x, cam_ch)
                    if cam_info is None:
                        row.append(Image.new("RGB", (int(W), int(H))))
                        continue

                    K3 = np.asarray(cam_info["camera_intrinsics"], np.float32).reshape(3, 3)

                    l2i = _lidar2image_from_caminfo(cam_info, K3)
                    row.append(self._get_3dbox_image_from_gtline(x, (W, H), l2i))
                imgs.append(row)

            result["3dbox_images"] = imgs
            for t, x in enumerate(seq):
                for v, _ in enumerate(self.sensor_channels):
                    tok = self._img_token(x, v)
                    p = self._png_path("3dbox_images", tok)
                    lock = p + ".lock"
                    if os.path.isfile(p) and _try_open_png(p) is not None:
                        continue
                    _acquire_lock(lock, timeout=30, stale=120, sleep=0.02)
                    try:
                        if os.path.isfile(p) and _try_open_png(p) is not None:
                            continue
                        _safe_save_png(imgs[t][v], p)
                    finally:
                        _release_lock(lock)

        # ---- hdmap_images (nuplan) cached
        cached = []
        all_hit = True
        for t, x in enumerate(seq):
            row = []
            for v, _ in enumerate(self.sensor_channels):
                tok = self._img_token(x, v)
                p = self._png_path("hdmap_images", tok)
                img = _try_open_png(p) if os.path.isfile(p) else None
                if img is None:
                    all_hit = False
                row.append(img)
            cached.append(row)

        if all_hit:
            result["hdmap_images"] = cached
        else:
            imgs = []
            for t, x in enumerate(seq):
                row = []
                for v, cam_ch in enumerate(self.sensor_channels):
                    W, H = img_sizes[t][v]
                    cam_intr = cam_Ks[t][v][:3, :3]
                    row.append(self._get_hdmap_image(x, cam_ch, (W, H), cam_intr))
                imgs.append(row)

            result["hdmap_images"] = imgs
            for t, x in enumerate(seq):
                for v, _ in enumerate(self.sensor_channels):
                    tok = self._img_token(x, v)
                    p = self._png_path("hdmap_images", tok)
                    lock = p + ".lock"
                    if os.path.isfile(p) and _try_open_png(p) is not None:
                        continue
                    _acquire_lock(lock, timeout=30, stale=120, sleep=0.02)
                    try:
                        if os.path.isfile(p) and _try_open_png(p) is not None:
                            continue
                        _safe_save_png(imgs[t][v], p)
                    finally:
                        _release_lock(lock)
                        
        # ---- pts proj (nuplan) cached -> proj_depth/proj_sem/proj_clr (tensors)
        if self.projected_pc_settings:
            s = self.projected_pc_settings

            final_hw = s.get("final_hw", None)                 # (H,W) or None
            invalid  = float(s.get("invalid_depth", -300.0))
            splat    = s.get("splat", [(1e9, 0)])

            n_bins = int(s.get("depth_bins", 256))
            gamma  = float(s.get("log_gamma", 1.0))
            far_m  = float(s.get("radius", 50.0))
            mode   = str(s.get("depth_bin_mode", "log")).lower()
            n_cls  = int(s.get("n_actor_classes", 3))

            depth_dir = s.get("depth_dir", f"proj_depth_g{gamma}_b{n_bins}")
            sem_dir   = s.get("sem_dir",   "proj_sem")
            clr_dir   = s.get("clr_dir",   "proj_clr")

            # >>> NEW: depth vis
            save_vis  = bool(s.get("save_depth_vis", True))
            vis_dir   = s.get("vis_dir", f"proj_depth_vis_g{gamma}_b{n_bins}")

            ensure_cache_subdir(self.cache_root, depth_dir)
            ensure_cache_subdir(self.cache_root, sem_dir)
            ensure_cache_subdir(self.cache_root, clr_dir)
            if save_vis:
                ensure_cache_subdir(self.cache_root, vis_dir)

            proj_depth, proj_sem, proj_clr = [], [], []

            for t, info in enumerate(seq):
                pts_xyz, pts_sem, clr_xyz, clr_rgb = self._compose_points_for_frame(info)

                row_d, row_s, row_c = [], [], []
                for v, cam_ch in enumerate(self.sensor_channels):
                    tok = self._img_token(info, v)

                    p_d = os.path.join(self.cache_root, depth_dir, f"{tok}.png")  # u16 png (scaled bins)
                    p_s = os.path.join(self.cache_root, sem_dir,   f"{tok}.png")  # RGB (0/255 one-hot)
                    p_c = os.path.join(self.cache_root, clr_dir,   f"{tok}.png")  # RGB
                    p_v = os.path.join(self.cache_root, vis_dir,   f"{tok}.png") if save_vis else None  # <<< NEW

                    bins_u16 = _try_open_u16_png(p_d) if os.path.isfile(p_d) else None
                    sem_img  = _try_open_png(p_s)     if os.path.isfile(p_s) else None
                    clr_img  = _try_open_png(p_c)     if os.path.isfile(p_c) else None
                    vis_img  = (_try_open_png(p_v) if (save_vis and os.path.isfile(p_v)) else None)  # <<< NEW

                    need_vis = save_vis and (vis_img is None)
                    need = (bins_u16 is None) or (sem_img is None) or (clr_img is None) or need_vis

                    cam_info = self._get_cam_info(info, cam_ch)
                    if cam_info is None:
                        W, H = img_sizes[t][v]
                        H0, W0 = int(H), int(W)
                        if final_hw is not None:
                            H0, W0 = int(final_hw[0]), int(final_hw[1])
                        row_d.append(torch.zeros((H0, W0), dtype=torch.long))
                        row_s.append(torch.zeros((n_cls, H0, W0), dtype=torch.float32))
                        row_c.append(torch.zeros((3, H0, W0), dtype=torch.float32))
                        continue

                    if need:
                        lock = p_c + ".lock"
                        _acquire_lock(lock, timeout=30, stale=120, sleep=0.02)
                        try:
                            # re-check under lock
                            if self.do_cache and os.path.isfile(p_d):
                                tmp = _try_open_u16_png(p_d)
                                if tmp is not None:
                                    bins_u16 = tmp
                            if self.do_cache and os.path.isfile(p_s):
                                sem_img = _try_open_png(p_s) or sem_img
                            if self.do_cache and os.path.isfile(p_c):
                                clr_img = _try_open_png(p_c) or clr_img
                            if save_vis and self.do_cache and os.path.isfile(p_v):
                                vis_img = _try_open_png(p_v) or vis_img

                            need_vis = save_vis and (vis_img is None)
                            need = (bins_u16 is None) or (sem_img is None) or (clr_img is None) or need_vis

                            if need:
                                W, H = img_sizes[t][v]
                                ori_hw = (int(H), int(W))

                                K3 = np.asarray(cam_info["camera_intrinsics"], np.float32).reshape(3, 3)
                                l2i = _lidar2image_from_caminfo(cam_info, K3)

                                recompute_main = (bins_u16 is None) or (sem_img is None) or (clr_img is None)

                                if recompute_main:
                                    depth, sem, clr = project_depth_sem_clr(
                                        pts_xyz, pts_sem, clr_xyz, clr_rgb,
                                        l2i, ori_hw,
                                        invalid_depth=invalid,
                                        splat=splat,
                                        n_actor_classes=n_cls
                                    )

                                    if final_hw is not None:
                                        depth = downsample_depth_blockwise(depth, final_hw, invalid=invalid)
                                        sem   = cv2.resize((sem * 255).astype(np.uint8),
                                                        (final_hw[1], final_hw[0]),
                                                        interpolation=cv2.INTER_NEAREST)
                                        sem   = (sem > 127).astype(np.uint8)
                                        clr   = downsample_clr_blockwise(clr, final_hw)

                                    # depth -> bins(u16)
                                    if mode in ("linear", "lin", "abs"):
                                        bins = depth_to_linbins_u16(depth, invalid=invalid, n_bins=n_bins, far_m=far_m)
                                    else:
                                        bins = depth_to_logbins_u16(depth, invalid=invalid, n_bins=n_bins, far_m=far_m, gamma=gamma)

                                    scale = max(1, 65535 // int(n_bins))
                                    bins_cache = (bins.astype(np.uint32) * scale).clip(0, 65535).astype(np.uint16)

                                    sem_rgb = (sem * 255).astype(np.uint8)   # (H,W,K)->(H,W,3)
                                    clr_u8  = clr.astype(np.uint8)

                                    if self.do_cache:
                                        _safe_save_u16_png(bins_cache, p_d)
                                        _safe_save_png(Image.fromarray(sem_rgb, "RGB"), p_s)
                                        _safe_save_png(Image.fromarray(clr_u8,  "RGB"), p_c)

                                    bins_u16 = bins_cache
                                    sem_img  = Image.fromarray(sem_rgb, "RGB")
                                    clr_img  = Image.fromarray(clr_u8,  "RGB")

                                    # <<< NEW: write vis from fresh `bins` (0..n_bins-1)
                                    if save_vis and (vis_img is None):
                                        vis_bgr = visualize_bins_u16(bins.astype(np.uint16), n_bins=n_bins, colormap=cv2.COLORMAP_TURBO)
                                        vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
                                        vis_img = Image.fromarray(vis_rgb, "RGB")
                                        if self.do_cache:
                                            _safe_save_png(vis_img, p_v)

                                else:
                                    if save_vis and (vis_img is None) and (bins_u16 is not None):
                                        bin_id = u16cache_to_binid(bins_u16, n_bins).astype(np.uint16)
                                        vis_bgr = visualize_bins_u16(bin_id, n_bins=n_bins, colormap=cv2.COLORMAP_TURBO)
                                        vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
                                        vis_img = Image.fromarray(vis_rgb, "RGB")
                                        if self.do_cache:
                                            _safe_save_png(vis_img, p_v)

                        finally:
                            _release_lock(lock)

                    # --- to tensors ---
                    bin_id = u16cache_to_binid(bins_u16, n_bins)  # (H,W)
                    row_d.append(torch.from_numpy(bin_id).long())

                    sem_np = (np.asarray(sem_img, np.uint8) > 127).astype(np.float32)
                    row_s.append(torch.from_numpy(sem_np).permute(2, 0, 1))

                    clr_np = np.asarray(clr_img, np.uint8).copy()
                    row_c.append(torch.from_numpy(clr_np).permute(2, 0, 1).float() / 255.)

                proj_depth.append(torch.stack(row_d, 0))
                proj_sem.append(torch.stack(row_s, 0))
                proj_clr.append(torch.stack(row_c, 0))

            result["proj_depth"] = torch.stack(proj_depth, 0)
            result["proj_sem"]   = torch.stack(proj_sem,   0)
            result["proj_clr"]   = torch.stack(proj_clr,   0)     
               
        # camera_transforms: ego_from_camera
        if "camera_transforms" not in result:
            cam_T = []
            for row_infos in cam_infos:  # [V]
                row_T = []
                for ci in row_infos:
                    if ci is None:
                        row_T.append(np.eye(4, dtype=np.float32))
                    else:
                        row_T.append(_se3_from_qt(ci["sensor2ego_rotation"], ci["sensor2ego_translation"]))
                cam_T.append(row_T)
            result["camera_transforms"] = torch.tensor(np.asarray(cam_T), dtype=torch.float32)  # [T,V,4,4]

        # ego_transforms: world_from_ego (same per view in a frame)
        if "ego_transforms" not in result:
            ego_T = []
            V = len(self.sensor_channels)
            for x in seq:
                world_from_ego = np.asarray(x[self.ego2global_key], np.float32)
                ego_T.append([world_from_ego] * V)
            result["ego_transforms"] = torch.tensor(np.asarray(ego_T), dtype=torch.float32)  # [T,V,4,4]

        if "lidar_transforms" not in result:
            T = len(seq)
            I4 = torch.eye(4, dtype=torch.float32)
            result["lidar_transforms"] = I4.view(1, 1, 4, 4).repeat(T, 1, 1, 1)  # [T,1,4,4]
            
        dwm.datasets.common.add_stub_key_data(self.stub_key_data_dict, result)
        
        if "clip_text" not in result:
            T = len(seq)
            V = len(self.sensor_channels)

            db_name = seq[0].get("db_name", None)

            cap = self.text_anno.get(db_name, {}) or {}
            text_main = make_image_description_string(cap, self.text_settings)

            tail = f"This is a nuplan video clip from {location}" if location else "This is a nuplan video clip"
            text = f"{text_main}. {tail}" if text_main else tail

            result["clip_text"] = [[text] * V for _ in range(T)]
                
        return result
