import io
import os
import pickle
from typing import Dict, List, Optional, Sequence, Tuple

import fsspec
import numpy as np
from PIL import Image, ImageDraw
import torch

import dwm.datasets.common

try:
    from scipy.spatial.transform import Rotation as _SciRotation
except Exception:  # pragma: no cover
    _SciRotation = None

try:
    import transforms3d.quaternions as _t3d_quat  # pragma: no cover
except Exception:  # pragma: no cover
    _t3d_quat = None


DEFAULT_CAMERA_NAMES = [
    "cam01", "cam03", "cam05", "cam06", "cam07", "cam08", "cam09"
]

# ONCE official labels -> a normalized set that is easy to reuse inside OpenDWM.
NAME_NORMALIZATION = {
    "Car": "car",
    "Truck": "truck",
    "Bus": "bus",
    "Pedestrian": "pedestrian",
    "Cyclist": "cyclist",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "pedestrian": "pedestrian",
    "cyclist": "cyclist",
}

DEFAULT_3DBOX_COLOR_TABLE = {
    "pedestrian": (255, 0, 0),
    "cyclist": (0, 255, 0),
    "car": (0, 0, 255),
    "truck": (0, 128, 255),
    "bus": (128, 0, 255),
}

_BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (0, 4), (1, 5), (2, 6), (3, 7),
    (4, 5), (5, 6), (6, 7), (7, 4),
]


def _rotation_matrix_from_xyzw(quaternion_xyzw: Sequence[float]) -> np.ndarray:
    """Convert ONCE pose quaternion (x, y, z, w) to a 3x3 rotation matrix."""
    q = np.asarray(quaternion_xyzw, dtype=np.float32).reshape(4)
    if _SciRotation is not None:
        return _SciRotation.from_quat(q).as_matrix().astype(np.float32)
    if _t3d_quat is None:  # pragma: no cover
        raise ImportError(
            "Need either scipy or transforms3d to convert ONCE quaternions."
        )
    # transforms3d expects wxyz
    wxyz = np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)
    return _t3d_quat.quat2mat(wxyz).astype(np.float32)


def _pose_matrix_from_xyzw_xyz(pose: Sequence[float]) -> np.ndarray:
    """ONCE pose format: (qx, qy, qz, qw, tx, ty, tz)."""
    pose = np.asarray(pose, dtype=np.float32).reshape(7)
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = _rotation_matrix_from_xyzw(pose[:4])
    mat[:3, 3] = pose[4:]
    return mat


def _normalize_name(name: str) -> str:
    return NAME_NORMALIZATION.get(str(name), str(name).lower())


def _make_image_description_string(
    caption_dict: Dict[str, str],
    settings: Optional[dict],
    random_state=None,
) -> str:
    settings = settings or {}
    rng = random_state if random_state is not None else np.random.default_rng()

    keys = list(settings.get("selected_keys", ["time", "weather"]))
    if settings.get("reorder_keys", False) and len(keys) > 1:
        keys = [keys[i] for i in rng.permutation(len(keys))]

    drop_rates = settings.get("drop_rates") or {}
    out = []
    for key in keys:
        value = caption_dict.get(key, None)
        if value in [None, ""]:
            continue
        if key in drop_rates and rng.random() <= float(drop_rates[key]):
            continue
        out.append(str(value))
    return ". ".join(out)


def _yaw_to_rotation(yaw: float) -> np.ndarray:
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)


def _box_corners_lidar(box7: np.ndarray) -> np.ndarray:
    """
    ONCE boxes_3d use center-based boxes in LiDAR coordinates:
    (cx, cy, cz, l, w, h, yaw).
    """
    cx, cy, cz, l, w, h, yaw = np.asarray(box7, dtype=np.float32).reshape(7)
    x = np.array([ l / 2,  l / 2, -l / 2, -l / 2,  l / 2,  l / 2, -l / 2, -l / 2], dtype=np.float32)
    y = np.array([ w / 2, -w / 2, -w / 2,  w / 2,  w / 2, -w / 2, -w / 2,  w / 2], dtype=np.float32)
    z = np.array([-h / 2, -h / 2, -h / 2, -h / 2,  h / 2,  h / 2,  h / 2,  h / 2], dtype=np.float32)
    corners = np.stack([x, y, z], axis=0)
    corners = (_yaw_to_rotation(yaw) @ corners).T
    corners += np.array([cx, cy, cz], dtype=np.float32)[None, :]
    return corners


def _project_lidar_points_to_image(
    points_lidar: np.ndarray,
    cam_to_velo: np.ndarray,
    intrinsic: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        points_lidar: (N, 3)
        cam_to_velo: 4x4 transform from camera to LiDAR.
        intrinsic: 3x3 camera intrinsics.

    Returns:
        uv: (N, 2) image coordinates
        depth: (N,) camera-frame depth
    """
    lidar_to_cam = np.linalg.inv(np.asarray(cam_to_velo, dtype=np.float32).reshape(4, 4))
    points_h = np.concatenate([
        np.asarray(points_lidar, dtype=np.float32),
        np.ones((points_lidar.shape[0], 1), dtype=np.float32)
    ], axis=1)
    cam = points_h @ lidar_to_cam.T
    xyz = cam[:, :3]
    depth = xyz[:, 2]
    uvw = xyz @ np.asarray(intrinsic, dtype=np.float32).reshape(3, 3).T
    uv = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-8, None)
    return uv, depth


def _safe_color_tuple(value, default=(0, 0, 0)) -> Tuple[int, int, int]:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        x = int(value)
        return (x, x, x)
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return tuple(int(i) for i in value)
    return default


class MotionDataset(torch.utils.data.Dataset):
    """
    OpenDWM-style ONCE dataset.

    Design choices for this implementation:
    1) No cache files.
    2) No projected point-cloud conditions.
    3) No distortion output, because the released ONCE camera images are already undistorted.
    4) The LiDAR frame is treated as the ego frame for camera_transforms
       because ONCE publishes cam_to_velo and per-frame LiDAR pose.
    """

    def __init__(
        self,
        fs: Optional[fsspec.AbstractFileSystem] = None,
        dataset_root: Optional[str] = None,
        info_pkl_path: Optional[str] = None,
        sequence_length: int = 1,
        fps_stride_tuples: Sequence[Tuple[float, float]] = (),
        sensor_channels: Sequence[str] = DEFAULT_CAMERA_NAMES,
        scene_key: str = "scene_id",
        timestamp_key: str = "timestamp",
        token_key: str = "token",
        enable_synchronization_check: bool = True,
        max_time_error_ratio: float = 0.5,
        max_time_error_us: Optional[int] = None,
        enable_camera_transforms: bool = True,
        enable_ego_transforms: bool = True,
        enable_sample_data: bool = False,
        enable_lidar_points: bool = False,
        _3dbox_image_settings: Optional[dict] = None,
        hdmap_image_settings: Optional[dict] = None,
        image_description_settings: Optional[dict] = None,
        stub_key_data_dict: Optional[dict] = None,
        return_clip_text: bool = False,
    ):
        if info_pkl_path is None:
            raise ValueError("info_pkl_path is required.")
        if fs is None:
            if dataset_root is None:
                raise ValueError("Either fs or dataset_root must be provided.")
            fs = fsspec.filesystem("file")

        self.fs = fs
        self._dataset_root = None if dataset_root is None else dataset_root.rstrip("/\\")
        self.info_pkl_path = info_pkl_path
        self.sequence_length = int(sequence_length)
        self.fps_stride_tuples = list(fps_stride_tuples)
        self.sensor_channels = list(sensor_channels)
        self.scene_key = scene_key
        self.timestamp_key = timestamp_key
        self.token_key = token_key
        self.enable_synchronization_check = bool(enable_synchronization_check)
        self.max_time_error_ratio = float(max_time_error_ratio)
        self.max_time_error_us = None if max_time_error_us is None else int(max_time_error_us)
        self.enable_camera_transforms = bool(enable_camera_transforms)
        self.enable_ego_transforms = bool(enable_ego_transforms)
        self.enable_sample_data = bool(enable_sample_data)
        self.enable_lidar_points = bool(enable_lidar_points)
        self.stub_key_data_dict = stub_key_data_dict
        self.return_clip_text = bool(return_clip_text)

        self._3dbox_image_settings = _3dbox_image_settings
        self.hdmap_image_settings = hdmap_image_settings
        self.image_description_settings = image_description_settings or {}
        self._image_description_rng = np.random.default_rng(
            self.image_description_settings.get("seed", None)
        )

        box_settings = _3dbox_image_settings or {}
        self._3dbox_pen_width = int(box_settings.get("pen_width", 4))
        self._3dbox_color_table = dict(DEFAULT_3DBOX_COLOR_TABLE)
        self._3dbox_color_table.update(box_settings.get("color_table", {}))
        self._fake_hdmap_color = _safe_color_tuple(
            (self.hdmap_image_settings or {}).get(
                "fake_condition_image_color",
                (self.hdmap_image_settings or {}).get("fake_color", (0, 0, 0)),
            ),
            default=(0, 0, 0)
        )

        infos = self._load_infos(self.info_pkl_path)
        self.scenes, self.scene_timestamps = self._build_scenes(infos)
        self.items = self._build_items()

    @staticmethod
    def _load_infos(info_pkl_path: str) -> List[dict]:
        with open(info_pkl_path, "rb") as f:
            obj = pickle.load(f)
        infos = obj["infos"] if isinstance(obj, dict) and "infos" in obj else obj
        if not isinstance(infos, list):
            raise TypeError("ONCE info pkl must be list[dict] or {'infos': list[dict]}.")
        return infos

    def _join_path(self, rel_path: str) -> str:
        if rel_path is None:
            return rel_path
        if os.path.isabs(rel_path):
            return rel_path
        if self._dataset_root is None:
            return rel_path
        strip_chars = "/\\"
        root = self._dataset_root.rstrip(strip_chars)
        rel = rel_path.lstrip(strip_chars)
        if root == "":
            return rel
        return f"{root}/{rel}"

    def _exists(self, rel_path: Optional[str]) -> bool:
        if not rel_path:
            return False
        try:
            return self.fs.exists(self._join_path(rel_path))
        except Exception:
            return False

    def _open_image(self, rel_path: Optional[str]) -> Optional[Image.Image]:
        if not rel_path:
            return None
        try:
            with self.fs.open(self._join_path(rel_path), "rb") as f:
                img = Image.open(io.BytesIO(f.read())).convert("RGB")
                img.load()
                return img
        except Exception:
            return None

    def _open_lidar_points(self, rel_path: Optional[str]) -> Optional[np.ndarray]:
        if not rel_path:
            return None
        try:
            with self.fs.open(self._join_path(rel_path), "rb") as f:
                data = np.frombuffer(f.read(), dtype=np.float32)
            if data.size % 4 != 0:
                return None
            return data.reshape(-1, 4)
        except Exception:
            return None

    @staticmethod
    def _default_image_size(info: dict) -> Tuple[int, int]:
        size = info.get("image_size", None)
        if size is None:
            size = (info.get("meta_info", {}) or {}).get("image_size", None)
        if size is None or len(size) != 2:
            return (1920, 1080)
        w, h = size
        return int(w), int(h)

    def _make_blank_image(self, info: dict, color=(0, 0, 0)) -> Image.Image:
        w, h = self._default_image_size(info)
        return Image.new("RGB", (w, h), tuple(int(i) for i in color))

    def _get_camera_info(self, info: dict, camera_name: str) -> Optional[dict]:
        return (info.get("cam", {}) or {}).get(camera_name, None)

    def _get_sensor_path(self, info: dict, camera_name: str) -> Optional[str]:
        return (info.get("image_paths", {}) or {}).get(camera_name, None)

    def _get_lidar_path(self, info: dict) -> Optional[str]:
        return info.get("lidar_path", None)

    def _build_scenes(self, infos: List[dict]):
        scenes = {}
        for info in infos:
            scene = info.get(self.scene_key, None)
            timestamp = info.get(self.timestamp_key, None)
            if scene is None or timestamp is None:
                continue
            scenes.setdefault(scene, []).append(info)

        scene_timestamps = {}
        out_scenes = {}
        for scene, scene_infos in scenes.items():
            scene_infos = sorted(scene_infos, key=lambda x: int(x[self.timestamp_key]))
            if len(scene_infos) < self.sequence_length:
                continue
            out_scenes[scene] = scene_infos
            scene_timestamps[scene] = np.asarray(
                [int(i[self.timestamp_key]) for i in scene_infos], dtype=np.int64
            )

        return out_scenes, scene_timestamps

    def _build_items(self) -> List[dict]:
        items = []
        for scene, ts in self.scene_timestamps.items():
            for fps, stride in self.fps_stride_tuples:
                items.extend(self._enumerate_segments(scene, ts, fps, stride))
        return items

    @staticmethod
    def _infer_timestamp_units_per_second(ts: np.ndarray) -> int:
        ts = np.asarray(ts, dtype=np.int64)
        if ts.size < 2:
            return 1000

        diffs = np.diff(ts)
        diffs = diffs[diffs > 0]
        if diffs.size == 0:
            return 1000

        median_dt = int(np.median(diffs))
        if median_dt >= 10000:
            return 1000000
        return 1000

    def _enumerate_segments(
        self,
        scene: str,
        ts: np.ndarray,
        fps: float,
        stride: float,
    ) -> List[dict]:
        T = self.sequence_length
        N = len(ts)
        items = []

        if T <= 0 or N < T:
            return items

        if float(fps) == 0.0:
            step = max(1, int(stride))
            for start in range(0, N - T + 1, step):
                items.append({
                    "scene": scene,
                    "fps": 0.0,
                    "indices": list(range(start, start + T)),
                    "sampling_mode": "index",
                })
            return items

        fps = float(fps)
        units_per_second = self._infer_timestamp_units_per_second(ts)
        dt = int(round(units_per_second / fps))
        seq_duration = (T - 1) * dt

        t_begin = int(ts[0])
        t_last_begin = int(ts[-1] - seq_duration)
        stride_units = dt if float(stride) <= 0 else int(round(float(stride) * units_per_second))

        if self.max_time_error_us is not None:
            max_err = int(round(self.max_time_error_us * units_per_second / 1000000))
        else:
            max_err = int(self.max_time_error_ratio * dt)

        max_err = max(1, max_err)
        stride_units = max(1, stride_units)

        if t_last_begin >= t_begin:
            t = t_begin
            while t <= t_last_begin:
                wanted = np.asarray([t + i * dt for i in range(T)], dtype=np.int64)
                picked_indices = [dwm.datasets.common.find_nearest(ts, int(w)) for w in wanted]

                if len(set(int(i) for i in picked_indices)) != T:
                    t += stride_units
                    continue

                if self.enable_synchronization_check:
                    picked_ts = ts[np.asarray(picked_indices, dtype=np.int64)]
                    if int(np.max(np.abs(picked_ts - wanted))) > max_err:
                        t += stride_units
                        continue

                items.append({
                    "scene": scene,
                    "fps": fps,
                    "indices": [int(i) for i in picked_indices],
                    "sampling_mode": "timestamp",
                    "timestamp_units_per_second": int(units_per_second),
                })
                t += stride_units

        if len(items) > 0:
            return items

        if N < 2:
            return items

        diffs = np.diff(ts.astype(np.int64))
        diffs = diffs[diffs > 0]
        if diffs.size == 0:
            return items

        median_dt = int(np.median(diffs))
        if median_dt <= 0:
            return items

        frame_step = max(1, int(round(dt / median_dt)))
        start_step = max(1, int(round(stride_units / median_dt)))
        last_start = N - (T - 1) * frame_step
        if last_start <= 0:
            return items

        interval_target = median_dt * frame_step
        interval_tolerance = max(max_err, int(1.5 * median_dt))

        for start in range(0, last_start, start_step):
            indices = [start + i * frame_step for i in range(T)]

            if self.enable_synchronization_check and T > 1:
                picked_ts = ts[np.asarray(indices, dtype=np.int64)]
                picked_diffs = np.diff(picked_ts)
                max_gap_error = int(np.max(np.abs(picked_diffs - interval_target)))
                if max_gap_error > interval_tolerance:
                    continue

            items.append({
                "scene": scene,
                "fps": fps,
                "indices": [int(i) for i in indices],
                "sampling_mode": "index_fallback",
                "estimated_source_fps": float(units_per_second / median_dt),
                "timestamp_units_per_second": int(units_per_second),
            })

        return items

    def __len__(self) -> int:
        return len(self.items)

    def _build_lidar_to_image(self, cam_info: dict) -> Tuple[np.ndarray, np.ndarray]:
        cam_to_velo = np.asarray(cam_info["cam_to_velo"], dtype=np.float32).reshape(4, 4)
        intrinsic = np.asarray(cam_info["camera_intrinsics"], dtype=np.float32).reshape(3, 3)
        return cam_to_velo, intrinsic

    def _render_3dbox_image(
        self,
        info: dict,
        camera_name: str,
        image_size: Tuple[int, int],
    ) -> Image.Image:
        img = Image.new("RGB", (int(image_size[0]), int(image_size[1])))
        draw = ImageDraw.Draw(img)

        cam_info = self._get_camera_info(info, camera_name)
        if cam_info is None:
            return img

        boxes = np.asarray(info.get("gt_boxes", np.zeros((0, 7), dtype=np.float32)), dtype=np.float32).reshape(-1, 7)
        names = list(info.get("gt_names", []))
        raw_names = list(info.get("gt_names_raw", []))

        if boxes.shape[0] == 0:
            return img

        cam_to_velo, intrinsic = self._build_lidar_to_image(cam_info)
        width, height = int(image_size[0]), int(image_size[1])

        for i, box in enumerate(boxes):
            corners = _box_corners_lidar(box)
            uv, depth = _project_lidar_points_to_image(corners, cam_to_velo, intrinsic)

            if np.any(depth <= 1e-4):
                continue
            if (
                np.max(uv[:, 0]) < 0 or np.min(uv[:, 0]) > width
                or np.max(uv[:, 1]) < 0 or np.min(uv[:, 1]) > height
            ):
                continue

            name = names[i] if i < len(names) else (raw_names[i] if i < len(raw_names) else "car")
            name = _normalize_name(name)
            color = self._3dbox_color_table.get(name, (255, 255, 255))

            for a, b in _BOX_EDGES:
                draw.line(
                    (
                        float(uv[a, 0]), float(uv[a, 1]),
                        float(uv[b, 0]), float(uv[b, 1]),
                    ),
                    fill=tuple(int(c) for c in color),
                    width=self._3dbox_pen_width,
                )
        return img

    def _build_hdmap_image(self, info: dict, image_size: Tuple[int, int]) -> Image.Image:
        # ONCE does not publish HD map assets in the released dataset package.
        return Image.new(
            "RGB",
            (int(image_size[0]), int(image_size[1])),
            self._fake_hdmap_color,
        )

    def _build_image_description(self, info: dict) -> str:
        caption_dict = {
            "time": info.get("period", (info.get("meta_info", {}) or {}).get("period", "")),
            "weather": info.get("weather", (info.get("meta_info", {}) or {}).get("weather", "")),
        }
        text = _make_image_description_string(
            caption_dict,
            self.image_description_settings,
            random_state=self._image_description_rng,
        )
        return text if text else "This is an ONCE driving scene."

    def __getitem__(self, index: int) -> dict:
        item = self.items[index]
        scene = item["scene"]
        seq = [self.scenes[scene][i] for i in item["indices"]]
        fps = float(item["fps"])

        camera_names = list(self.sensor_channels)
        T = len(seq)
        V = len(camera_names)

        seq_ts = np.asarray([int(frame[self.timestamp_key]) for frame in seq], dtype=np.int64)
        timestamp_units_per_second = self._infer_timestamp_units_per_second(seq_ts)
        t0 = int(seq_ts[0])
        half_second = timestamp_units_per_second // 2
        pts = torch.tensor(
            [
                [float((int(ts_i) - t0 + half_second) // timestamp_units_per_second)] * V
                for ts_i in seq_ts
            ],
            dtype=torch.float32,
        )

        result = {
            "fps": torch.tensor(fps, dtype=torch.float32),
            "pts": pts,
        }

        if self.enable_sample_data:
            result["sample_data"] = seq
            result["scene"] = {"token": scene}

        images = []
        camera_intrinsics = []
        camera_transforms = []
        image_sizes = []

        for frame in seq:
            row_images = []
            row_intrinsics = []
            row_transforms = []
            row_sizes = []

            for camera_name in camera_names:
                cam_info = self._get_camera_info(frame, camera_name)
                image = self._open_image(self._get_sensor_path(frame, camera_name))
                if image is None:
                    image = self._make_blank_image(frame)

                row_images.append(image)
                row_sizes.append((image.width, image.height))

                if cam_info is None:
                    row_intrinsics.append(np.eye(3, dtype=np.float32))
                    row_transforms.append(np.eye(4, dtype=np.float32))
                else:
                    row_intrinsics.append(
                        np.asarray(cam_info["camera_intrinsics"], dtype=np.float32).reshape(3, 3)
                    )
                    row_transforms.append(
                        np.asarray(cam_info["cam_to_velo"], dtype=np.float32).reshape(4, 4)
                    )

            images.append(row_images)
            camera_intrinsics.append(row_intrinsics)
            camera_transforms.append(row_transforms)
            image_sizes.append(row_sizes)

        result["images"] = images

        if self.enable_camera_transforms:
            result["camera_intrinsics"] = torch.tensor(
                np.asarray(camera_intrinsics), dtype=torch.float32
            )
            result["camera_transforms"] = torch.tensor(
                np.asarray(camera_transforms), dtype=torch.float32
            )
            result["image_size"] = torch.tensor(
                np.asarray(image_sizes, dtype=np.int64), dtype=torch.long
            )

        if self.enable_ego_transforms:
            ego_transforms = []
            for frame in seq:
                ego = np.asarray(frame["ego2global"], dtype=np.float32).reshape(4, 4)
                ego_transforms.append([ego] * V)
            result["ego_transforms"] = torch.tensor(
                np.asarray(ego_transforms), dtype=torch.float32
            )

        if self.enable_lidar_points:
            lidar_points = []
            for frame in seq:
                pts4 = self._open_lidar_points(self._get_lidar_path(frame))
                if pts4 is None:
                    pts_xyz = np.zeros((0, 3), dtype=np.float32)
                else:
                    pts_xyz = np.asarray(pts4[:, :3], dtype=np.float32)
                lidar_points.append(torch.tensor(pts_xyz, dtype=torch.float32))
            result["lidar_points"] = lidar_points
            result["lidar_transforms"] = (
                torch.eye(4, dtype=torch.float32)
                .view(1, 1, 4, 4)
                .repeat(T, 1, 1, 1)
            )

        if self._3dbox_image_settings is not None:
            images_3dbox = []
            for t, frame in enumerate(seq):
                row = []
                for v, camera_name in enumerate(camera_names):
                    row.append(self._render_3dbox_image(frame, camera_name, image_sizes[t][v]))
                images_3dbox.append(row)
            result["3dbox_images"] = images_3dbox

        if self.hdmap_image_settings is not None:
            hdmap_images = []
            for t, frame in enumerate(seq):
                row = []
                for v in range(V):
                    row.append(self._build_hdmap_image(frame, image_sizes[t][v]))
                hdmap_images.append(row)
            result["hdmap_images"] = hdmap_images

        descriptions = []
        for frame in seq:
            text = self._build_image_description(frame)
            descriptions.append([text] * V)

        result["image_description"] = descriptions
        if self.return_clip_text:
            result["clip_text"] = descriptions

        dwm.datasets.common.add_stub_key_data(self.stub_key_data_dict, result)
        return result
