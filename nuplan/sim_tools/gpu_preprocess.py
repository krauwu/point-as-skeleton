"""
"""
import os
import numpy as np
import torch
import pickle


class GPUPointPreprocessor:
    """

    """

    def __init__(self, proj_cfg, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.cfg = proj_cfg

        self.bg_scene_gpu = {}
        cs = proj_cfg.get("color_scene_by_location", {}) or {}
        for k, npy_path in cs.items():
            bg = np.asarray(np.load(npy_path, allow_pickle=False))
            xyz = bg[:, :3].astype(np.float32)
            rgb = np.clip(bg[:, 3:6], 0, 255).astype(np.uint8)

            xyz_gpu = torch.from_numpy(xyz).float().to(self.device)
            rgb_gpu = torch.from_numpy(rgb).to(self.device)

            self.bg_scene_gpu[k] = {'xyz': xyz_gpu, 'rgb': rgb_gpu}

        self.actor_root = proj_cfg.get("actor_root", None)
        tpl_root = proj_cfg.get("actor_template_root", None)
        self.actor_tpl = {}

        if tpl_root:
            for fn in os.listdir(tpl_root):
                if fn.endswith(".pkl"):
                    with open(os.path.join(tpl_root, fn), "rb") as f:
                        tpl = pickle.load(f)
                        self.actor_tpl[fn[:-4]] = tpl

        for k in list(self.actor_tpl):
            tpl = self.actor_tpl[k]
            if tpl is not None:
                tpl_tensor = torch.from_numpy(
                    np.asarray(tpl).astype(np.float32)
                ).to(self.device)
                self.actor_tpl[k] = tpl_tensor

    def compose_points_gpu(self, info: dict, radius=150.0):
        """

            pts_xyz_gpu: (N,3) float32 on GPU
            pts_sem_gpu: (N,) int32 on GPU
            clr_xyz_gpu: (M,3) float32 on GPU
            clr_rgb_gpu: (M,3) uint8 on GPU
        """
        # Parse ego2global
        ego2global = torch.from_numpy(
            np.asarray(info["ego2global"], np.float32)
        ).to(self.device)

        R = ego2global[:3, :3]  # ego->global
        t = ego2global[:3, 3]

        # ---------- 1) Background (GPU) ----------
        bg_data = self.bg_scene_gpu.get(info.get("db_name"), None)
        if bg_data is None:
            bg_xyz_l = torch.zeros((0, 3), dtype=torch.float32, device=self.device)
            bg_rgb = torch.zeros((0, 3), dtype=torch.uint8, device=self.device)
        else:
            xyz_g = bg_data['xyz'].clone()
            rgb = bg_data['rgb']

            mask = (xyz_g[:, 0] > t[0] - radius) & \
                   (xyz_g[:, 0] < t[0] + radius) & \
                   (xyz_g[:, 1] > t[1] - radius) & \
                   (xyz_g[:, 1] < t[1] + radius)

            xyz_g = xyz_g[mask]
            rgb = rgb[mask]

            xyz_g = xyz_g - t.unsqueeze(0)
            bg_xyz_l = torch.mm(xyz_g, R)  # (N,3) = (N,3) @ (3,3)
            bg_rgb = rgb

        bg_sem = torch.zeros((bg_xyz_l.shape[0],), dtype=torch.long, device=self.device)

        # ---------- 2) Actors ----------
        boxes = info.get("gt_boxes", None)
        if boxes is None:
            boxes = []
        boxes = np.asarray(boxes, dtype=np.float32)

        names = _as_list(info.get("gt_names", None))
        tracks = info.get("track_token", None)
        tracks = _as_list(tracks)
        if len(tracks) == 0:
            tracks = [None] * len(boxes)

        name2id = {"car": 1, "ped": 2, "bike": 3}
        min_actor_pts = int(self.cfg.get("min_actor_points", 30000))

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
            if tok is not None and self.actor_root is not None:
                p = os.path.join(self.actor_root, f"{tok}.npy")
                if os.path.isfile(p):
                    track = np.asarray(np.load(p))
                    if track.shape[0] > 0:
                        actor_track = torch.from_numpy(
                            track.astype(np.float32)
                        ).to(self.device)

            sem_source = None

            if actor_track is not None and actor_track.shape[0] >= min_actor_pts:
                sem_source = actor_track[:, :3]
            else:
                if cls == "car":
                    actor_tpl = self.actor_tpl.get("suv")
                else:
                    actor_tpl = self.actor_tpl.get(cls)

                if actor_tpl is None:
                    continue

                sem_source = actor_tpl[:, :3]

            xyz_sem = sem_source

            # actor_tpl = self.actor_tpl.get(cls)
            # if actor_tpl is None and cls == "car":
            #     actor_tpl = _pick_car_template_gpu(self.actor_tpl, dz)

            # if actor_tpl is None:
            #     continue

            # xyz_sem = actor_tpl[:, :3]

            Rz = _yaw_to_Rz_gpu(yaw, device=self.device)
            xyz_sem = torch.mm(xyz_sem, Rz.T) + torch.tensor([x, y, z], device=self.device).unsqueeze(0)

            act_xyz_list.append(xyz_sem)
            act_sem_list.append(torch.full((xyz_sem.shape[0],), sid, dtype=torch.long, device=self.device))

            if actor_track is not None and actor_track.shape[0] > min_actor_pts and actor_track.shape[1] >= 6:
                xyz_clr = actor_track[:, :3]
                xyz_clr = torch.mm(xyz_clr, Rz.T) + torch.tensor([x, y, z], device=self.device).unsqueeze(0)

                a_rgb = actor_track[:, 3:6]

                # mx = torch.nanmax(a_rgb) # require torch>=1.9.0
                mx = torch.max(torch.nan_to_num(a_rgb, nan=float('-inf')))
                if mx <= 1.5:
                    a_rgb_u8 = torch.clamp(a_rgb * 255.0, 0, 255).byte()
                else:
                    a_rgb_u8 = torch.clamp(a_rgb, 0, 255).byte()

                act_clr_xyz_list.append(xyz_clr)
                act_clr_rgb_list.append(a_rgb_u8)

        # Concatenate on GPU
        if act_xyz_list:
            act_xyz = torch.cat(act_xyz_list, 0)
            act_sem = torch.cat(act_sem_list, 0)
            act_clr_xyz = torch.cat(act_clr_xyz_list, 0) if act_clr_xyz_list else torch.zeros((0, 3), dtype=torch.float32, device=self.device)
            act_clr_rgb = torch.cat(act_clr_rgb_list, 0) if act_clr_rgb_list else torch.zeros((0, 3), dtype=torch.uint8, device=self.device)
        else:
            act_xyz = torch.zeros((0, 3), dtype=torch.float32, device=self.device)
            act_sem = torch.zeros((0,), dtype=torch.long, device=self.device)
            act_clr_xyz = torch.zeros((0, 3), dtype=torch.float32, device=self.device)
            act_clr_rgb = torch.zeros((0, 3), dtype=torch.uint8, device=self.device)

        pts_xyz_gpu = torch.cat([bg_xyz_l, act_xyz], 0)
        pts_sem_gpu = torch.cat([bg_sem, act_sem], 0)

        clr_xyz_gpu = torch.cat([bg_xyz_l, act_clr_xyz], 0)
        clr_rgb_gpu = torch.cat([bg_rgb, act_clr_rgb], 0)

        return pts_xyz_gpu, pts_sem_gpu, clr_xyz_gpu, clr_rgb_gpu


# Helper functions
def _as_list(x):
    if x is None:
        return []
    return list(x) if isinstance(x, (list, tuple, np.ndarray)) else [x]


def _yaw_to_Rz_gpu(yaw, device='cpu'):
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    return torch.tensor([[c, -s, 0.0],
                        [s, c, 0.0],
                        [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)


def _pick_car_template_gpu(actor_tpl, dz, device='cpu'):
    dz = float(dz)
    if dz < 1.8:
        return actor_tpl.get("sedan")
    if dz < 2.05:
        return actor_tpl.get("suv")
    if dz < 2.7:
        return actor_tpl.get("pickup")
    return None


def get_gpu_preprocessor(proj_cfg, device='cuda'):
    return GPUPointPreprocessor(proj_cfg, device)
