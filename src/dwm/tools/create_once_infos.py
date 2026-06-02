import argparse
import json
import os
import pickle as pkl
from os import path as osp
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

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


def _rotation_matrix_from_xyzw(quaternion_xyzw: Sequence[float]) -> np.ndarray:
    q = np.asarray(quaternion_xyzw, dtype=np.float32).reshape(4)
    if _SciRotation is not None:
        return _SciRotation.from_quat(q).as_matrix().astype(np.float32)
    if _t3d_quat is None:  # pragma: no cover
        raise ImportError(
            "Need either scipy or transforms3d to convert ONCE quaternions."
        )
    wxyz = np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)
    return _t3d_quat.quat2mat(wxyz).astype(np.float32)


def pose_xyzw_xyz_to_matrix(pose: Sequence[float]) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32).reshape(7)
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = _rotation_matrix_from_xyzw(pose[:4])
    mat[:3, 3] = pose[4:]
    return mat


def normalize_name(name: str) -> str:
    return NAME_NORMALIZATION.get(str(name), str(name).lower())


def load_split_sequence_ids(dataset_root: str, split: str) -> List[str]:
    split_path = osp.join(dataset_root, "ImageSets", f"{split}.txt")
    if not osp.isfile(split_path):
        raise FileNotFoundError(f"Cannot find split file: {split_path}")
    with open(split_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _reshape_array(x, shape, dtype=np.float32):
    arr = np.asarray(x, dtype=dtype)
    return arr.reshape(shape)


def parse_sequence_infos(
    dataset_root: str,
    sequence_id: str,
    camera_names: Sequence[str] = DEFAULT_CAMERA_NAMES,
    require_all_cameras: bool = True,
    require_lidar: bool = True,
) -> List[dict]:
    seq_dir = osp.join(dataset_root, "data", sequence_id)
    anno_path = osp.join(seq_dir, f"{sequence_id}.json")
    if not osp.isfile(anno_path):
        return []

    with open(anno_path, "r", encoding="utf-8") as f:
        anno_json = json.load(f)

    meta_info = anno_json.get("meta_info", {}) or {}
    calib = anno_json.get("calib", {}) or {}
    frames = anno_json.get("frames", []) or []

    image_size = meta_info.get("image_size", [1920, 1080])
    camera_names = list(camera_names)

    infos = []
    for frame in frames:
        frame_id = str(frame["frame_id"])
        timestamp = int(frame_id)

        image_paths = {}
        missing_camera = False
        for cam in camera_names:
            rel = osp.join("data", sequence_id, cam, f"{frame_id}.jpg")
            abs_path = osp.join(dataset_root, rel)
            if osp.isfile(abs_path):
                image_paths[cam] = rel
            else:
                missing_camera = True

        lidar_rel = osp.join("data", sequence_id, "lidar_roof", f"{frame_id}.bin")
        lidar_abs = osp.join(dataset_root, lidar_rel)
        if require_lidar and (not osp.isfile(lidar_abs)):
            continue
        if require_all_cameras and missing_camera:
            continue

        cam_dict: Dict[str, dict] = {}
        for cam in camera_names:
            if cam not in calib:
                continue
            cam_item = calib[cam]
            cam_dict[cam] = {
                "cam_to_velo": _reshape_array(cam_item["cam_to_velo"], (4, 4)),
                "camera_intrinsics": _reshape_array(cam_item["cam_intrinsic"], (3, 3)),
            }

        annos = frame.get("annos", None)
        if annos is None:
            gt_boxes = np.zeros((0, 7), dtype=np.float32)
            gt_names_raw = []
            gt_names = []
            boxes_2d = {}
            num_points_in_gt = []
        else:
            boxes_3d = annos.get("boxes_3d", [])
            gt_boxes = (
                np.asarray(boxes_3d, dtype=np.float32).reshape(-1, 7)
                if len(boxes_3d) > 0
                else np.zeros((0, 7), dtype=np.float32)
            )
            gt_names_raw = [str(i) for i in annos.get("name", [])]
            gt_names = [normalize_name(i) for i in gt_names_raw]
            boxes_2d = annos.get("boxes_2d", {}) or {}
            num_points_in_gt = [int(i) for i in annos.get("num_points_in_gt", [])]

        pose = np.asarray(frame["pose"], dtype=np.float32).reshape(7)

        infos.append({
            "scene_id": sequence_id,
            "scene_token": sequence_id,
            "token": f"{sequence_id}|{frame_id}",
            "frame_id": frame_id,
            "timestamp": timestamp,
            "sequence_id": frame.get("sequence_id", sequence_id),
            "pose_xyzw_xyz": pose,
            "ego2global": pose_xyzw_xyz_to_matrix(pose),
            "meta_info": meta_info,
            "weather": meta_info.get("weather", ""),
            "period": meta_info.get("period", ""),
            "image_size": [int(image_size[0]), int(image_size[1])],
            "cam": cam_dict,
            "image_paths": image_paths,
            "lidar_path": lidar_rel if osp.isfile(lidar_abs) else None,
            "gt_boxes": gt_boxes,
            "gt_names": gt_names,
            "gt_names_raw": gt_names_raw,
            "boxes_2d": boxes_2d,
            "num_points_in_gt": num_points_in_gt,
        })

    return infos


def collect_once_infos(
    dataset_root: str,
    split: str,
    camera_names: Sequence[str] = DEFAULT_CAMERA_NAMES,
    require_all_cameras: bool = True,
    require_lidar: bool = True,
    sequence_ids: Optional[Sequence[str]] = None,
) -> List[dict]:
    if sequence_ids is None:
        sequence_ids = load_split_sequence_ids(dataset_root, split)

    all_infos = []
    for seq_id in sequence_ids:
        seq_infos = parse_sequence_infos(
            dataset_root=dataset_root,
            sequence_id=seq_id,
            camera_names=camera_names,
            require_all_cameras=require_all_cameras,
            require_lidar=require_lidar,
        )
        all_infos.extend(seq_infos)
    return all_infos


def create_once_infos(
    dataset_root: str,
    split: str,
    output_path: str,
    camera_names: Sequence[str] = DEFAULT_CAMERA_NAMES,
    require_all_cameras: bool = True,
    require_lidar: bool = True,
    sequence_ids: Optional[Sequence[str]] = None,
):
    infos = collect_once_infos(
        dataset_root=dataset_root,
        split=split,
        camera_names=camera_names,
        require_all_cameras=require_all_cameras,
        require_lidar=require_lidar,
        sequence_ids=sequence_ids,
    )

    metadata = {
        "dataset": "ONCE",
        "split": split,
        "camera_names": list(camera_names),
        "require_all_cameras": bool(require_all_cameras),
        "require_lidar": bool(require_lidar),
    }

    out_obj = {"infos": infos, "metadata": metadata}
    os.makedirs(osp.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        pkl.dump(out_obj, f)

    print(f"saved {len(infos)} frames to {output_path}")


def build_argparser():
    parser = argparse.ArgumentParser(description="Create OpenDWM ONCE info pkl.")
    parser.add_argument("--dataset-root", required=True, help="Root directory of ONCE dataset.")
    parser.add_argument("--split", required=True,
                        choices=["train", "val", "test", "raw_small", "raw_medium", "raw_large"])
    parser.add_argument("--output-path", required=True, help="Output pkl path.")
    parser.add_argument(
        "--camera-names",
        nargs="+",
        default=DEFAULT_CAMERA_NAMES,
        help="Camera channels to keep.",
    )
    parser.add_argument(
        "--allow-missing-cameras",
        action="store_true",
        help="Keep frames even if some requested camera images are missing.",
    )
    parser.add_argument(
        "--allow-missing-lidar",
        action="store_true",
        help="Keep frames even if lidar_roof/<frame>.bin is missing.",
    )
    parser.add_argument(
        "--sequence-ids",
        nargs="*",
        default=None,
        help="Optional explicit sequence ids. If unset, read ImageSets/<split>.txt.",
    )
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    create_once_infos(
        dataset_root=args.dataset_root,
        split=args.split,
        output_path=args.output_path,
        camera_names=args.camera_names,
        require_all_cameras=not args.allow_missing_cameras,
        require_lidar=not args.allow_missing_lidar,
        sequence_ids=args.sequence_ids,
    )


if __name__ == "__main__":
    main()
