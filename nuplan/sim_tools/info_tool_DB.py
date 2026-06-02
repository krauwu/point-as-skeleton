# nuplan_info_tool.py
import copy
import numpy as np
from pyquaternion import Quaternion

CAM_TYPES = ["CAM_L1","CAM_L0","CAM_F0","CAM_R0","CAM_R1","CAM_R2","CAM_B0","CAM_L2"]


def _wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _lidar_box_to_global(box_lidar, ego2global, lidar2ego, ego_yaw):
    """
    return: center_global(3,), yaw_global
    """
    x_l, y_l, z_l, L, W, H, yaw_l = box_lidar.tolist()

    p_l = np.array([x_l, y_l, z_l, 1.0], dtype=np.float32)
    p_ego = lidar2ego @ p_l
    p_g = ego2global @ p_ego
    center_g = p_g[:3]

    yaw_g = _wrap_pi(float(ego_yaw) + float(yaw_l))
    return center_g, yaw_g, (L, W, H)


def _global_to_lidar_box(center_g, yaw_g, ego2global_future, lidar2ego, ego_yaw_future, dims):
    """
    center_g: (3,)
    return box_lidar_future (7,)
    """
    L, W, H = dims

    g2ego = np.linalg.inv(ego2global_future)
    ego2lidar = np.linalg.inv(lidar2ego)

    p_g = np.array([center_g[0], center_g[1], center_g[2], 1.0], dtype=np.float32)
    p_ego = g2ego @ p_g
    p_l = ego2lidar @ p_ego

    x_l, y_l, z_l = float(p_l[0]), float(p_l[1]), float(p_l[2])
    yaw_l = _wrap_pi(float(yaw_g) - float(ego_yaw_future))
    return np.array([x_l, y_l, z_l, L, W, H, yaw_l], dtype=np.float32)


def fill_future_boxes_rollout_reactive_simple(
    infos_all,
    history_len=9,
    pack_hz=5,
    # --- stop / lock ---
    stop_speed_thresh=0.30,
    stop_disp_thresh=0.05,
    lock_stop_if_decelerating=True,

    # --- yaw / lateral keep ---
    yawrate_eps=1e-3,
    yawrate_decay=0.6,

    # --- collision / braking ---
    front_lat_margin=0.6,
    safe_gap=2.0,
    max_decel=6.0,
    max_speed=40.0,              # m/s
):
    """
    """

    dt = 1.0 / float(pack_hz)
    past = infos_all[:history_len]
    future = infos_all[history_len:]

    if history_len < 2:
        return infos_all
    if len(future) == 0:
        return infos_all

    info1 = past[-1]
    info0 = past[-2]

    if info0.get("gt_boxes", None) is None or info1.get("gt_boxes", None) is None:
        return infos_all
    if info0.get("track_token", None) is None or info1.get("track_token", None) is None:
        return infos_all

    lidar2ego = info1["lidar2ego"]
    ego2g0 = info0["ego2global"]
    ego2g1 = info1["ego2global"]
    yaw0 = float(info0["egopose"][3])
    yaw1 = float(info1["egopose"][3])

    tok0 = info0["track_token"]
    tok1 = info1["track_token"]
    box0 = info0["gt_boxes"]
    box1 = info1["gt_boxes"]
    name1 = info1["gt_names"]

    # ============== 1) global state map(t0/t1) ==============
    map0 = {}
    for i in range(len(tok0)):
        tk = tok0[i]
        if tk is None:
            continue
        c_g, yaw_g, dims = _lidar_box_to_global(box0[i], ego2g0, lidar2ego, yaw0)
        map0[tk] = (c_g, float(yaw_g), dims)

    map1 = {}
    for i in range(len(tok1)):
        tk = tok1[i]
        if tk is None:
            continue
        c_g, yaw_g, dims = _lidar_box_to_global(box1[i], ego2g1, lidar2ego, yaw1)
        map1[tk] = (c_g, float(yaw_g), dims, name1[i])

    common_tokens = [tk for tk in map1.keys() if tk in map0]
    if len(common_tokens) == 0:
        return infos_all

    have_prev = history_len >= 3 and past[-3].get("gt_boxes", None) is not None and past[-3].get("track_token", None) is not None
    map_prev = {}
    if have_prev:
        info_prev = past[-3]
        ego2g_prev = info_prev["ego2global"]
        yaw_prev = float(info_prev["egopose"][3])
        tok_prev = info_prev["track_token"]
        box_prev = info_prev["gt_boxes"]
        for i in range(len(tok_prev)):
            tk = tok_prev[i]
            if tk is None:
                continue
            c_g, yaw_g, dims = _lidar_box_to_global(box_prev[i], ego2g_prev, lidar2ego, yaw_prev)
            map_prev[tk] = (c_g, float(yaw_g), dims)

    states = {}
    for tk in common_tokens:
        c0, yaw_g0, dims0 = map0[tk]
        c1, yaw_g1, dims1, nm = map1[tk]

        disp = c1[:2] - c0[:2]
        speed = float(np.linalg.norm(disp) / max(dt, 1e-6))
        speed = float(np.clip(speed, 0.0, max_speed))

        dyaw = _wrap_pi(yaw_g1 - yaw_g0)
        yaw_rate = float(dyaw / max(dt, 1e-6))

        lock_stop = False
        if float(np.linalg.norm(disp)) < stop_disp_thresh and speed < stop_speed_thresh:
            lock_stop = True

        if have_prev and lock_stop_if_decelerating and (tk in map_prev):
            c_prev, yaw_prev_g, _ = map_prev[tk]
            disp_prev = c0[:2] - c_prev[:2]
            speed_prev = float(np.linalg.norm(disp_prev) / max(dt, 1e-6))
            if speed < stop_speed_thresh and speed <= speed_prev + 1e-3:
                lock_stop = True

        states[tk] = {
            "pos": c1.astype(np.float32),      # global xyz
            "yaw": float(yaw_g1),
            "v": float(speed),
            "yaw_rate": float(yaw_rate),
            "dims": dims1,
            "name": nm,
            "lock_stop": bool(lock_stop),
        }

    # ============== 4) rollout step-by-step ==============
    for k in range(len(future)):
        info_f = future[k]
        ego2g_f = info_f["ego2global"]
        ego_yaw_f = float(info_f["egopose"][3])

        next_pred = {}
        for tk, st in states.items():
            if st["lock_stop"] or st["v"] < stop_speed_thresh:
                next_pred[tk] = (st["pos"].copy(), st["yaw"], 0.0, 0.0)
                continue

            v = float(st["v"])
            yaw = float(st["yaw"])
            yaw_rate = float(st["yaw_rate"])

            yaw_rate = float(yaw_rate * yawrate_decay)

            if abs(yaw_rate) < yawrate_eps:
                dx = v * dt * np.cos(yaw)
                dy = v * dt * np.sin(yaw)
                yaw_next = yaw
            else:
                yaw_next = _wrap_pi(yaw + yaw_rate * dt)
                dx = (v / yaw_rate) * (np.sin(yaw_next) - np.sin(yaw))
                dy = (v / yaw_rate) * (-np.cos(yaw_next) + np.cos(yaw))

            pos_next = st["pos"].copy()
            pos_next[0] += float(dx)
            pos_next[1] += float(dy)

            next_pred[tk] = (pos_next, yaw_next, v, yaw_rate)

        for tk, st in states.items():
            if st["lock_stop"]:
                st["v"] = 0.0
                st["yaw_rate"] = 0.0
                continue

            pos_i, yaw_i, v_i, yawrate_i = next_pred[tk]

            # heading unit
            hx = float(np.cos(yaw_i))
            hy = float(np.sin(yaw_i))

            Li, Wi, Hi = st["dims"]
            ri = 0.5 * float(max(Wi, 1.0))

            best_front_s = None

            for tj, stj in states.items():
                if tj == tk:
                    continue
                pos_j = next_pred[tj][0]
                Lj, Wj, Hj = stj["dims"]
                rj = 0.5 * float(max(Wj, 1.0))

                rel = pos_j[:2] - pos_i[:2]
                s_front = float(rel[0] * hx + rel[1] * hy)
                d_lat = float(-rel[0] * hy + rel[1] * hx)

                if s_front <= 0.0:
                    continue
                if abs(d_lat) > (0.5 * Wi + 0.5 * Wj + front_lat_margin):
                    continue

                gap = s_front - (ri + rj) - safe_gap
                if best_front_s is None or gap < best_front_s:
                    best_front_s = gap

            if best_front_s is None:
                st["pos"] = pos_i
                st["yaw"] = yaw_i
                st["v"] = float(np.clip(v_i, 0.0, max_speed))
                st["yaw_rate"] = yawrate_i
                continue

            gap = float(best_front_s)

            if gap <= 0.0:
                st["pos"] = st["pos"]
                st["yaw"] = yaw_i
                st["v"] = 0.0
                st["yaw_rate"] = 0.0
                st["lock_stop"] = True
                continue

            a_req = - (v_i * v_i) / (2.0 * max(gap, 1e-3))
            a = float(max(a_req, -max_decel))

            v_new = float(max(0.0, v_i + a * dt))

            if v_new < stop_speed_thresh:
                st["pos"] = st["pos"]
                st["yaw"] = yaw_i
                st["v"] = 0.0
                st["yaw_rate"] = 0.0
                st["lock_stop"] = True
            else:
                pos_new = st["pos"].copy()
                pos_new[0] += v_new * dt * np.cos(st["yaw"])
                pos_new[1] += v_new * dt * np.sin(st["yaw"])

                st["pos"] = pos_new
                st["yaw"] = yaw_i
                st["v"] = v_new
                st["yaw_rate"] = yawrate_i

        pred_boxes = []
        pred_names = []
        pred_tokens = []

        for tk, st in states.items():
            c_pred = st["pos"]
            yaw_pred = st["yaw"]
            dims = st["dims"]
            nm = st["name"]

            box_l = _global_to_lidar_box(
                center_g=c_pred,
                yaw_g=yaw_pred,
                ego2global_future=ego2g_f,
                lidar2ego=lidar2ego,
                ego_yaw_future=ego_yaw_f,
                dims=dims,
            )
            pred_boxes.append(box_l)
            pred_names.append(nm)
            pred_tokens.append(tk)

        info_f["gt_boxes"] = np.stack(pred_boxes, axis=0).astype(np.float32) if len(pred_boxes) > 0 \
                             else np.zeros((0, 7), dtype=np.float32)
        info_f["gt_names"] = np.array(pred_names, dtype=object)
        info_f["track_token"] = np.array(pred_tokens, dtype=object)

    return infos_all



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


def _quat_to_wxyz(q):
    """
    """
    if hasattr(q, "w") and hasattr(q, "x"):
        return np.array([q.w, q.x, q.y, q.z], dtype=np.float32)
    q = np.asarray(q, dtype=np.float32).reshape(-1)
    if q.size == 4:
        return q
    raise ValueError(f"Invalid quaternion shape: {q.shape}")


def build_cam_dict_from_db_record(db_record, ego2global_4x4):
    """
      camera_intrinsics(3x3), distortion, sensor2ego_translation, sensor2ego_rotation(wxyz)
    """
    # lidar record
    lid_record = db_record.lidar[0]

    t = ego2global_4x4[:3, 3].astype(np.float32)
    R = np.asarray(ego2global_4x4[:3, :3], dtype=np.float64)

    ego_pose_like = type("EgoPoseLike", (), {})()
    ego_pose_like.translation_np = t
    ego_pose_like.quaternion = Quaternion(matrix=R)

    cam_dict = {}
    for cam_ch in CAM_TYPES:
        cam_db = db_record.camera.select_one(channel=cam_ch)
        cam_info = obtain_sensor2top(cam_db, lid_record, ego_pose_like)

        K3 = np.asarray(cam_info["camera_intrinsics"], np.float32).reshape(3, 3)
        dist = np.asarray(cam_info.get("distortion", []), np.float32).reshape(-1)

        cam_dict[cam_ch] = {
            "camera_intrinsics": K3,
            "distortion": dist,
            "sensor2ego_translation": np.asarray(cam_info["sensor2ego_translation"], np.float32).reshape(3),
            "sensor2ego_rotation": _quat_to_wxyz(cam_info["sensor2ego_rotation"]),
            "sensor2lidar_rotation": np.asarray(cam_info["sensor2lidar_rotation"], np.float32).reshape(3, 3),
            "sensor2lidar_translation": np.asarray(cam_info["sensor2lidar_translation"], np.float32).reshape(3),
        }
    return cam_dict


def build_info_from_sim_state(ego_state, tracked_objects, lidar2ego, trans_z=0.0,
                              db_name=None, location=None,
                              db_record=None):
    """
      - egopose: [x,y,z,yaw]
      - ego2global: 4x4
      - gt_names: str
      - track_token: str/bytes
    """
    info = {}

    # ===== timestamp =====
    info["timestamp"] = int(ego_state.time_us)

    # ===== egopose =====
    ego_x = float(ego_state.rear_axle.x)
    ego_y = float(ego_state.rear_axle.y)
    ego_yaw = float(ego_state.rear_axle.heading)
    info["egopose"] = [ego_x, ego_y, float(trans_z), ego_yaw]

    # ===== ego2global =====
    try:
        ego2global = ego_state.rear_axle.as_matrix_3d()
        ego2global[2, 3] = float(trans_z)
        info["ego2global"] = ego2global
    except Exception:
        info["ego2global"] = None

    info["lidar2ego"] = lidar2ego

    # ===== ego_feats =====
    try:
        v = ego_state.dynamic_car_state.rear_axle_velocity_2d.array
        a = ego_state.dynamic_car_state.rear_axle_acceleration_2d.array
        w = float(ego_state.dynamic_car_state.angular_velocity)
        info["ego_feats"] = np.array([v[0], v[1], a[0], a[1], w], dtype=np.float32)
    except Exception:
        info["ego_feats"] = None

    # ===== scene/meta for pipeline =====
    info["db_name"] = db_name
    info["location"] = location

    # ===== cam calib for pipeline =====
    info["cam"] = build_cam_dict_from_db_record(db_record, ego2global)
    gt_boxes = []
    gt_names = []
    track_tokens = []

    if tracked_objects is not None and lidar2ego is not None:
        ego2lidar = np.linalg.inv(lidar2ego)

        c = np.cos(-ego_yaw)
        s = np.sin(-ego_yaw)

        for obj in tracked_objects:
            if obj.box is None:
                continue

            # --- obj global ---
            ox = float(obj.center.x)
            oy = float(obj.center.y)
            oyaw = float(obj.center.heading)

            L = float(obj.box.length)
            W = float(obj.box.width)
            H = float(obj.box.height)

            # --- global -> ego(2D) ---
            dx = ox - ego_x
            dy = oy - ego_y
            x_ego = c * dx - s * dy
            y_ego = s * dx + c * dy
            z_ego = 0.0 + 0.5 * H

            # --- ego -> lidar ---
            p_ego = np.array([x_ego, y_ego, z_ego, 1.0], dtype=np.float32)
            p_l = ego2lidar @ p_ego
            x_l, y_l, z_l = float(p_l[0]), float(p_l[1]), float(p_l[2])

            yaw_l = oyaw - ego_yaw
            yaw_l = (yaw_l + np.pi) % (2 * np.pi) - np.pi

            if L < 1e-2 or W < 1e-2:
                continue

            gt_boxes.append([x_l, y_l, z_l, L, W, H, yaw_l])

            t = obj.tracked_object_type.name
            if t == "VEHICLE":
                name = "car"
            elif t == "PEDESTRIAN":
                name = "ped"
            elif t == "BICYCLE":
                name = "bike"
            else:
                continue
            gt_names.append(name)

            try:
                track_tokens.append(obj.metadata.track_token)
            except Exception:
                track_tokens.append(None)

    info["gt_boxes"] = np.array(gt_boxes, dtype=np.float32)
    info["gt_names"] = np.array(gt_names)
    info["track_token"] = np.array(track_tokens)

    return info


# ==========================
# ==========================

class RollingInfoBuffer5Hz:
    """
    """

    def __init__(
            self,
            history_len=9,
            future_len=10,
            sim_hz=10,
            pack_hz=5,
            lidar_hz=20,
        ):
        self.history_len = history_len
        self.future_len = future_len

        self.sim_hz = int(sim_hz)
        self.pack_hz = int(pack_hz)
        self.lidar_hz = int(lidar_hz)

        assert self.sim_hz % self.pack_hz == 0, f"sim_hz={self.sim_hz} must be divisible by pack_hz={self.pack_hz}"
        assert self.lidar_hz % self.pack_hz == 0, f"lidar_hz={self.lidar_hz} must be divisible by pack_hz={self.pack_hz}"
        assert self.lidar_hz % self.sim_hz == 0, f"lidar_hz={self.lidar_hz} must be divisible by sim_hz={self.sim_hz}"

        self.stride_sim = self.sim_hz // self.pack_hz

        self.stride_lidar = self.lidar_hz // self.pack_hz

        self.lidar_per_sim = self.lidar_hz // self.sim_hz

        self.lidar2ego = None
        self.cam_db_dict = None

        self.trans_z = 0.0

        self._scenario = None
        self._iter_idx = 0

    def set_db_context(self, scenario, iter_idx: int):
        self._scenario = scenario
        self._iter_idx = int(iter_idx)

    def _get_db_obs(self, it: int):
        if self._scenario is None:
            return None
        return self._scenario.get_tracked_objects_at_iteration(int(it))

        
    def set_static_metas(self, lidar2ego: np.ndarray, cam_db_dict=None, trans_z: float = 0.0):
        self.lidar2ego = lidar2ego
        self.cam_db_dict = cam_db_dict
        self.trans_z = float(trans_z) 

        
    def build_infos(self, planner_input, future_trajectory, past_ego_states=None):
        """
        """
        hist = planner_input.history
        ego_buf = past_ego_states if past_ego_states is not None else hist.ego_state_buffer

        past_infos = []
        for i in range(self.history_len):
            if past_ego_states is not None:
                ego_state_i = ego_buf[i]
            else:
                buf_idx = -1 - (self.history_len - 1 - i) * self.stride_sim
                if -buf_idx > len(ego_buf):
                    buf_idx = -len(ego_buf)
                ego_state_i = ego_buf[buf_idx]

            it = self._iter_idx - (self.history_len - 1 - i) * self.stride_sim
            if it < 0:
                it = 0

            obs_i = self._get_db_obs(it)
            tracked_i = obs_i.tracked_objects if obs_i is not None else None
            tracked_list = tracked_i.tracked_objects if tracked_i is not None else None

            info_i = build_info_from_sim_state(
                ego_state=ego_state_i,
                tracked_objects=tracked_list,
                lidar2ego=self.lidar2ego,
                trans_z=self.trans_z,
                db_name=self.db_name,
                location=self.location,
                db_record=self.db_record,
            )
            past_infos.append(info_i)

        if len(past_infos) == 0:
            assert len(past_infos) > 0, (
                f"[build_infos] past_infos empty! "
                f"stride_sim={self.stride_sim} history_len={self.history_len}"
            )

        future_infos = []

        if future_trajectory is not None:
            sampled_states = future_trajectory.get_sampled_trajectory()

            # pack_hz=5 -> stride=2；pack_hz=2 -> stride=5
            future_traj_stride = int(10 / self.pack_hz)
            start_i = future_traj_stride
            end_i = start_i + self.future_len * future_traj_stride

            needed = []
            for j in range(start_i, min(end_i, len(sampled_states)), future_traj_stride):
                needed.append(sampled_states[j])

            if len(needed) > self.future_len:
                needed = needed[:self.future_len]

            template_info = past_infos[-1]
            future_dt_s = 1.0 / self.pack_hz

            for k, st in enumerate(needed):
                it_f = self._iter_idx + (k + 1) * self.stride_sim
                obs_f = self._get_db_obs(it_f)
                tracked_f = obs_f.tracked_objects if obs_f is not None else None
                tracked_f_list = tracked_f.tracked_objects if tracked_f is not None else None

                info_f = build_info_from_sim_state(
                    ego_state=st,
                    tracked_objects=tracked_f_list,
                    lidar2ego=self.lidar2ego,
                    trans_z=self.trans_z,
                    db_name=self.db_name,
                    location=self.location,
                    db_record=self.db_record,
                )
                future_infos.append(info_f)

        return past_infos + future_infos
